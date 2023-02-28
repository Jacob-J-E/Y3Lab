


class Generating_res_data:

    def __init__(self,params_e1,params_e2) -> None:
        self.params_e1 = params_e1
        self.params_e2 = params_e2

    def gen_res_data(self,):
        params_e1 = self.params_e1
        params_e2 = self.params_e2
        bounds = (params_e2[1] - params_e1[1])/4