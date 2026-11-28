"""Tests for piazzi.print_progress."""

import io
import math

import pytest
import torch

import piazzi

D = torch.float64
t = torch.linspace(0, 1, 20, dtype=D)
y = 2 * torch.exp(-3 * t)


def residuals(p):
    return p[0] * torch.exp(-p[1] * t) - y


def cost(p):
    return 0.5 * residuals(p).square().sum()


def rows(text):
    lines = text.strip().splitlines()
    return lines[0], [line.split() for line in lines[1:]]


def test_rows_match_the_iterations():
    out, infos = io.StringIO(), []
    printer = piazzi.print_progress(file=out)

    def callback(info):
        infos.append(info)
        printer(info)

    piazzi.least_squares(residuals, torch.tensor([1.0, 1.0], dtype=D), callback=callback)
    header, body = rows(out.getvalue())
    assert header.split() == ["iter", "fun", "rel_decrease", "step", "n_fun_grad", "n_fun", "n_system"]
    assert len(body) == len(infos)
    for row, info in zip(body, infos, strict=True):
        assert int(row[0]) == info.iteration
        assert float(row[1]) == pytest.approx(info.fun.item(), rel=1e-6)
        assert float(row[2]) == pytest.approx((info.fun_start - info.fun).item() / info.fun_start.item(), rel=1e-3)
        assert row[3] == ("accepted" if info.accepted else "rejected")
        assert [int(v) for v in row[4:]] == [info.n_fun_grad, info.n_fun, info.n_system]


def rosenbrock(x):
    return torch.stack([10 * (x[1] - x[0] ** 2), 1 - x[0]])


def test_every_and_rejected_steps():
    # On the Rosenbrock residuals from the classic start, Levenberg-Marquardt rejects some steps.
    x0 = torch.tensor([-1.2, 1.0], dtype=D)
    out = io.StringIO()
    piazzi.least_squares(rosenbrock, x0, callback=piazzi.print_progress(file=out))
    _, body = rows(out.getvalue())
    assert {r[3] for r in body} == {"accepted", "rejected"}
    out = io.StringIO()
    piazzi.least_squares(rosenbrock, x0, callback=piazzi.print_progress(every=3, file=out))
    _, every3 = rows(out.getvalue())
    assert [int(r[0]) for r in every3] == [int(r[0]) for r in body if int(r[0]) % 3 == 0]


def test_header_on_each_run():
    out = io.StringIO()
    printer = piazzi.print_progress(file=out)
    piazzi.least_squares(residuals, torch.tensor([1.0, 1.0], dtype=D), callback=printer)
    piazzi.minimize(cost, torch.tensor([1.0, 1.0], dtype=D), callback=printer)
    assert out.getvalue().count("rel_decrease") == 2


def test_prints_to_stdout_by_default(capsys):
    piazzi.lbfgs_armijo(cost, torch.tensor([1.0, 1.0], dtype=D), max_iter=3, callback=piazzi.print_progress())
    header, body = rows(capsys.readouterr().out)
    assert header.split()[0] == "iter" and len(body) == 3
    assert all(int(r[4]) > 0 for r in body)  # L-BFGS counts value-and-gradient evaluations


def test_torch_optim_and_zero_start():
    out = io.StringIO()
    printer = piazzi.print_progress(file=out)
    piazzi.minimize(
        cost, torch.tensor([2.0, 3.0], dtype=D), method=torch.optim.Adam, options={"max_iter": 2}, callback=printer
    )
    _, body = rows(out.getvalue())
    assert len(body) == 2
    # At a zero objective the relative decrease is undefined: printed as nan, not an error.
    info = piazzi.IterationInfo(0, torch.zeros(2), torch.tensor(0.0), torch.tensor(0.0), None, True)
    out = io.StringIO()
    piazzi.print_progress(file=out)(info)
    assert math.isnan(float(rows(out.getvalue())[1][0][2]))


@pytest.mark.parametrize("every", [0, -1, 1.5])
def test_invalid_every(every):
    with pytest.raises(ValueError, match="every"):
        piazzi.print_progress(every=every)
