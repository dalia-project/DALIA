import sys

from gar1.itest import gar1_itest
from gar2.itest import gar2_itest
from gr.itest import gr_itest
from gs.itest import gs_itest
from gst.itest import gst_itest
from gstcoreg2.itest import gstcoreg2_itest
from gstsmall.itest import gstsmall_itest
from par1.itest import par1_itest
from pr.itest import pr_itest
from pst.itest import pst_itest

# run_test_scripts = {
#     "gst/itest.py": ["seq", "par_f", "par_s"],
#     "gcoreg/itest.py": ["seq", "par_f", "par_s"],
#     "par1/itest.py": ["seq", "par_f"],
#     "pr/itest.py": ["seq"],
# }

itest_calls = {
    gst_itest: ["seq"],
    gstcoreg2_itest: ["seq"],
    par1_itest: ["seq"],
    pr_itest: ["seq"],
    gr_itest: ["seq"],
    gar1_itest: ["seq"],
    gar2_itest: ["seq"],
    gs_itest: ["seq"],
    gstsmall_itest: ["seq"],
    pst_itest: ["seq"],
}

# The backend is selected through the environment, before DALIA gets imported:
#   ARRAY_MODULE=numpy python runner.py
#   ARRAY_MODULE=cupy mpirun -n 2 python runner.py

if __name__ == "__main__":

    failed = []
    for itest, modes in itest_calls.items():
        result = itest()
        print(f"{itest.__name__} in mode `{modes[0]}` returned: {result}", flush=True)

        # The number of iterations is informative, it does not fail the test
        if not result.startswith(("success", "warning")):
            failed.append(itest.__name__)

    if failed:
        print(f"Failed integration tests: {', '.join(failed)}", flush=True)
        sys.exit(1)
