import efs_production_generator as pm


def main():
    nhb_pm = pm.NhbProductionModel(
        import_home=r'Y:\NorMITs Demand\import',
        export_home=r'Y:\NorMITs Demand\norms_2015\v2_3-EFS_Output\iter1',
        model_name='norms',
    )
    nhb_pm.run()


if __name__ == '__main__':
    main()
