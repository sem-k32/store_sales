import torch
from .lambda_fun import lambdaLinear, lambdaLinearSpline
from typing import Union


class smoothSplineProjector:
    def __init__(self, lambda_spline, smoothness: int = 1) -> None:
        """make splines satisfy smooth conditions in the end points

        Args:
            smoothness (int, optional): smoothness of the splines in the end points. Defaults to 1.
        """
        self._lambda_spline = lambda_spline
        self._smooth = smoothness

        # build projection matrix

        # form supplementary rows
        pows_row = torch.empty(0)
        last_day_row = torch.empty(0)
        for last_day in self._lambda_spline._days_per_month:
            # days start with zero
            last_day -= 1

            pow_subrow = [last_day ** i for i in range(self._lambda_spline._poly_ord, -1, -1)]
            pow_subrow[-1] = 0

            last_day_subrow = torch.ones(self._lambda_spline._poly_ord + 1) * last_day

            pows_row = torch.concat([pows_row, torch.FloatTensor(pow_subrow)], dim=0)
            last_day_row = torch.concat([last_day_row, last_day_subrow], dim=0)

        diff_row = torch.arange(self._lambda_spline._poly_ord, 0 - 1, -1).repeat(12)

        # build constraint matrix
        constr_matrix = torch.empty((self._lambda_spline._smooth + 1, pows_row.shape[0]), dtype=torch.float64)
        for i in range(self._lambda_spline._smooth + 1):
            constr_matrix[i] = pows_row

            pows_row /= last_day_row
            pows_row *= diff_row
            pows_row[self._lambda_spline._poly_ord - 1 - i::self._lambda_spline._poly_ord + 1] = 0

            diff_row -= 1

        # debug
        self._constraint_matrix = constr_matrix

        # compute projection matrix
        temp = torch.matmul(torch.linalg.pinv(constr_matrix.T).T, constr_matrix)
        self._proj_matr = torch.eye(temp.shape[0]) - temp
        self._proj_matr = self._proj_matr.to(torch.float64)

    def Project(self) -> None:
        with torch.no_grad():
            # onpromotion projection
            if self._lambda_spline._promotion_c < 0:
                self._lambda_spline._promotion_c = 0

            # splines projection

            # form a vector out of polys params
            params = torch.concat(list(self._lambda_spline._polys))
            # perform projection
            params = torch.matmul(self._proj_matr, params)
            # update state dict
            state_dict = self._lambda_spline.state_dict()

            i = 0
            for key in state_dict.keys():
                if key != "_promotion_c":
                    state_dict[key] = params[
                        i * (self._lambda_spline._poly_ord + 1): (i + 1) * (self._lambda_spline._poly_ord + 1)
                    ]
                    i += 1

            self._lambda_spline.load_state_dict(state_dict)

            # debug
            # test, that constraints on polys hold
            temp = torch.matmul(self._constraint_matrix, params)
            if not torch.allclose(temp, torch.zeros_like(temp), atol=1e-3):
                raise ValueError("Constraints violation")


class posCoefsProjector:
    def __init__(self, lambda_func: Union[lambdaLinearSpline, lambdaLinear]) -> None:
        """ make linear model's coefs positive. That must be enough for lambda's positivity
        """
        self._lambda_func = lambda_func

    def Project(self) -> None:
        state_dict = self._lambda_func.state_dict()

        for key in state_dict.keys():
            # promotion and polys coefs projection
            state_dict[key] *= (state_dict[key] >= 0)

        self._lambda_func.load_state_dict(state_dict)



