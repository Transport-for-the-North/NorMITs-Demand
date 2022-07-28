import normits_demand as nd
from normits_demand import constants as consts
from normits_demand.logging import initialise_package_logger


def _validate_constants() -> None:
    """
    Checks for any errors in the constants of normits_demand
    """
    # BACKLOG: Update the NoRMS Post-ME decompile process in EFS to match the
    #  new compilation process naming
    #  labels: EFS, NoRMS
    return
    # Check the NORMS matrices all match
    post_me_keys = list(consts.NORMS_VDM_SEG_TO_NORMS_POSTME_NAMING.keys())
    pre_me_keys = (list(consts.NORMS_VDM_SEG_INTERNAL.keys())
                   + list(consts.NORMS_VDM_SEG_EXTERNAL.keys()))

    if post_me_keys != pre_me_keys:
        raise nd.InitialisationError(
            "Error while checking constants! The keys for the NoRMS pre and "
            "post ME conversions don't match. This WILL break the compilation "
            "and decompilation processes"
        )


def _initialise():
    _validate_constants()
    initialise_package_logger()

