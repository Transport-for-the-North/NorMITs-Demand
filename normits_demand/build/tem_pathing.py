import os

import normits_demand.build.pathing as pathing


class TEMPathing(pathing.Pathing):

    def __init__(
            self,
            config_path,
            param_file):
        super().__init__(config_path,
                         param_file)

        self.export = self._check_exports()

    def _check_exports(self):

        """
        """

        output_dir = os.path.join(self.run_folder,
                                  self.params['iteration'])
        p_output_f = 'Production Outputs'
        a_output_f = 'Attraction Outputs'

        p_in_hb = os.path.join(
            output_dir,
            p_output_f,
            'hb_productions_' +
            self.params['land_use_zoning'].lower() +
            '.csv')

        a_in_hb = os.path.join(
            output_dir,
            a_output_f,
            'hb_attractions_' +
            self.params['land_use_zoning'].lower() +
            '.csv')

        p_in_nhb = os.path.join(
            output_dir,
            p_output_f,
            'nhb_productions_' +
            self.params['land_use_zoning'] +
            '.csv')

        a_in_nhb = os.path.join(
            output_dir,
            a_output_f,
            'nhb_attractions_' +
            self.params['land_use_zoning'] +
            '.csv')

        p_out_hb = os.path.join(
            output_dir,
            p_output_f,
            'hb_productions_' +
            self.params['model_zoning'].lower() +
            '.csv')

        a_out_hb = os.path.join(
            output_dir,
            a_output_f,
            'hb_attractions_' +
            self.params['model_zoning'].lower() +
            '.csv')

        p_out_nhb = os.path.join(
            output_dir,
            p_output_f,
            'nhb_productions_' +
            self.params['model_zoning'].lower() +
            '.csv')

        a_out_nhb = os.path.join(
            output_dir,
            a_output_f,
            'nhb_attractions_' +
            self.params['model_zoning'].lower() +
            '.csv')

        if not os.path.exists(p_in_hb):
            in_hb = ''
        if not os.path.exists(p_in_nhb):
            in_nhb = ''
        if not os.path.exists(p_out_hb):
            out_hb = ''
        if not os.path.exists(p_out_nhb):
            out_nhb = ''

        export_dict = {'p_in_hb': p_in_hb,
                       'a_in_hb': a_in_hb,
                       'p_in_nhb': p_in_nhb,
                       'a_in_nhb': a_in_nhb,
                       'p_out_hb': p_out_hb,
                       'a_out_hb': a_out_hb,
                       'p_out_nhb': p_out_nhb,
                       'a_out_nhb': a_out_nhb}

        self.export = export_dict

        return export_dict

