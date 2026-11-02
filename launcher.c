#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <mach-o/dyld.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <sys/stat.h>

int main(int argc, char *argv[]) {
    // 1. Resolve absolute path of the executable
    char exe_path[PATH_MAX];
    uint32_t size = sizeof(exe_path);
    if (_NSGetExecutablePath(exe_path, &size) != 0) {
        fprintf(stderr, "GigaDictate: Unable to resolve executable path\n");
        return 1;
    }

    char resolved_exe[PATH_MAX];
    if (realpath(exe_path, resolved_exe) == NULL) {
        strncpy(resolved_exe, exe_path, sizeof(resolved_exe) - 1);
        resolved_exe[sizeof(resolved_exe) - 1] = '\0';
    }

    // 2. Derive bundle directory hierarchy
    // resolved_exe: <AppDir>/Contents/MacOS/GigaDictate
    char contents_dir[PATH_MAX];
    strncpy(contents_dir, resolved_exe, sizeof(contents_dir) - 1);
    contents_dir[sizeof(contents_dir) - 1] = '\0';

    // Strip binary name (e.g. /GigaDictate)
    char *p = strrchr(contents_dir, '/');
    if (p) *p = '\0';
    // Strip MacOS directory (e.g. /MacOS)
    p = strrchr(contents_dir, '/');
    if (p) *p = '\0';

    char resources_dir[PATH_MAX];
    snprintf(resources_dir, sizeof(resources_dir), "%s/Resources", contents_dir);

    char frameworks_dir[PATH_MAX];
    snprintf(frameworks_dir, sizeof(frameworks_dir), "%s/Frameworks", contents_dir);

    // 3. Check if standalone bundled Python & venv exist inside the application bundle
    char bundled_py_home[PATH_MAX];
    snprintf(bundled_py_home, sizeof(bundled_py_home), "%s/Python.framework/Versions/3.11", frameworks_dir);

    char bundled_site[PATH_MAX];
    snprintf(bundled_site, sizeof(bundled_site), "%s/venv/lib/python3.11/site-packages", resources_dir);

    char bundled_pkg[PATH_MAX];
    snprintf(bundled_pkg, sizeof(bundled_pkg), "%s/giga_dictate", resources_dir);

    struct stat st;
    int is_bundled = (stat(bundled_pkg, &st) == 0);

    // 4. Configure environment
    setenv("PATH", "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin", 1);
    setenv("PYTHONUNBUFFERED", "1", 1);
    setenv("PYTHONDONTWRITEBYTECODE", "1", 1);

    if (is_bundled) {
        setenv("PYTHONHOME", bundled_py_home, 1);
        chdir(resources_dir);
    } else {
        // Fallback for running `./launcher` directly from local repo
        char repo_dir[PATH_MAX];
        strncpy(repo_dir, resolved_exe, sizeof(repo_dir) - 1);
        repo_dir[sizeof(repo_dir) - 1] = '\0';
        char *slash = strrchr(repo_dir, '/');
        if (slash) *slash = '\0';
        chdir(repo_dir);
    }

    // 5. Initialize embedded Python 3.11 runtime
    Py_Initialize();

    // 6. Bootstrap sys.path dynamically
    char bootstrap[4096];
    if (is_bundled) {
        snprintf(bootstrap, sizeof(bootstrap),
            "import sys, site, os\n"
            "res = '%s'\n"
            "site_dir = '%s'\n"
            "if os.path.exists(site_dir):\n"
            "    site.addsitedir(site_dir)\n"
            "gigaam = os.path.join(res, 'GigaAM_repo')\n"
            "for p in [res, gigaam]:\n"
            "    if os.path.exists(p) and p not in sys.path:\n"
            "        sys.path.insert(0, p)\n",
            resources_dir, bundled_site
        );
    } else {
        char repo_dir[PATH_MAX];
        strncpy(repo_dir, resolved_exe, sizeof(repo_dir) - 1);
        repo_dir[sizeof(repo_dir) - 1] = '\0';
        char *slash = strrchr(repo_dir, '/');
        if (slash) *slash = '\0';

        snprintf(bootstrap, sizeof(bootstrap),
            "import sys, site, os\n"
            "proj_dir = '%s'\n"
            "venv_dir = os.path.join(proj_dir, '.venv')\n"
            "site_packages = os.path.join(venv_dir, 'lib', 'python3.11', 'site-packages')\n"
            "if os.path.exists(site_packages):\n"
            "    site.addsitedir(site_packages)\n"
            "if proj_dir not in sys.path:\n"
            "    sys.path.insert(0, proj_dir)\n",
            repo_dir
        );
    }

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
