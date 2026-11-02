#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]) {
    // 1. Change working directory to project workspace
    const char *project_dir = "/Users/ctazzerc/Projects/Giga dictate";
    chdir(project_dir);

    // 2. Configure environment
    setenv("PATH", "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin", 1);
    setenv("VIRTUAL_ENV", "/Users/ctazzerc/Projects/Giga dictate/.venv", 1);
    setenv("PYTHONUNBUFFERED", "1", 1);

    // 3. Initialize embedded Python 3.11 runtime
    Py_Initialize();

    // 4. Bootstrap virtualenv site-packages
    const char *bootstrap =
        "import sys, site, os\n"
        "venv_dir = '/Users/ctazzerc/Projects/Giga dictate/.venv'\n"
        "site_packages = os.path.join(venv_dir, 'lib', 'python3.11', 'site-packages')\n"
        "site.addsitedir(site_packages)\n"
        "proj_dir = '/Users/ctazzerc/Projects/Giga dictate'\n"
        "if proj_dir not in sys.path:\n"
        "    sys.path.insert(0, proj_dir)\n";

    PyRun_SimpleString(bootstrap);

    // If custom arguments passed from terminal (and not macOS -psn launch parameter)
    if (argc > 1 && strncmp(argv[1], "-psn", 4) != 0) {
        return Py_BytesMain(argc, argv);
    }

    int ret = PyRun_SimpleString("import giga_dictate.main\ngiga_dictate.main.main()\n");
    if (ret != 0) {
        fprintf(stderr, "GigaDictate Python execution error: %d\n", ret);
    }

    Py_Finalize();
    return ret;
}
