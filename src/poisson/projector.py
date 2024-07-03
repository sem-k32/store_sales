import torch
from lambda_spline import lambdaSpline


class smoothSplineProjector:
    def __init__(self, lambda_spline: lambdaSpline) -> None:
        self._lambda_spline: lambdaSpline = lambda_spline

        # build projection matrix

        # form supplementary rows
        pows_row = torch.empty(0)
        last_day_row = torch.empty(0)
        for last_day in self._lambda_spline._days_per_month:
            # days start with zero
            last_day -= 1

            pow_subrow = [last_day ** i for i in range(self._lambda_spline._poly_ord, -1, -1)]
            pow_subrow[-1] = 0

            last_day_subrow = torch.ones(self._lambda_spline._poly_ord) * last_day

            pows_row = torch.concat([pows_row, torch.FloatTensor(pow_subrow)], dim=0)
            last_day_row = torch.concat([last_day_row, last_day_subrow], dim=0)

        diff_row = torch.arange(self._lambda_spline._poly_ord, 0 - 1, -1).expand(12)

        # build constraint matrix
        constr_matrix = torch.empty((self._lambda_spline._smooth, pows_row.shape[0]))
        for i in range(self._lambda_spline._smooth):
            constr_matrix[i] = pows_row

            pows_row /= last_day_row
            pows_row *= diff_row
            pows_row[self._lambda_spline._poly_ord - 1 - i::self._lambda_spline._poly_ord + 1] = 0

            diff_row -= 1

        # compute projection matrix
        temp = torch.matmul(torch.linalg.pinv(constr_matrix.T).T, constr_matrix)
        self._proj_matr = torch.eye(temp.shape[0]) - temp

    def Project(self) -> None:
        with torch.no_grad():
            # onpromotion projection
            if self._lambda_spline._promotion_c < 0:
                self._lambda_spline._promotion_c = 0

            # splines projection

            # form a vector out of polys params
            params = torch.empty(0)
            for param in self._lambda_spline._polys:
                params = torch.concat([params, param], dim=0)

            # perform projection
            params = torch.dot(self._proj_matr, params)



