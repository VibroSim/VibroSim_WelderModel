import sys
#from numpy.distutils.core import setup as numpy_setup, Extension as numpy_Extension
import os
import os.path
import subprocess
import re
from setuptools import setup
from setuptools.command.install_lib import install_lib
from setuptools.command.install import install
from setuptools.command.build_ext import build_ext
import setuptools.command.bdist_egg
import sys
from Cython.Build import cythonize


extra_compile_args = {
    "msvc": ["/openmp"],
    #"gcc": ["-O0", "-g", "-Wno-uninitialized"),    # Replace the line below with this line to enable debugging of the compiled extension
    "gcc": ["-fopenmp","-O5","-Wno-uninitialized"],
    "clang": ["-fopenmp","-O5","-Wno-uninitialized"],
}

extra_include_dirs = {
    "msvc": [".",np.get_include()],
    "gcc": [".",np.get_include()],
    "clang": [".",np.get_include()],
}

extra_libraries = {
    "msvc": [],
    "gcc": ["gomp",],
    "clang": [],
}

extra_link_args = {
    "msvc": [],
    "gcc": [],
    "clang": ["-fopenmp=libomp"],
}

def check_for_opencl(compiler,include_dirs,library_dirs):
    got_include=False
    got_library=False
    
    for include_dir in include_dirs:
        if os.path.exists(os.path.join(include_dir,"CL","cl.h")):
            got_include=True
            pass
        pass

    if compiler=="msvc" or compiler=="mingw":
        for library_dir in library_dirs:
            if os.path.exists(os.path.join(library_dir,"OpenCL.lib")):
                got_library=True
                pass
            pass

        pass
    else: 
        for library_dir in library_dirs:
            if os.path.exists(os.path.join(library_dir,"libOpenCL.so")):
                got_library=True
                pass
            pass
        pass

    return (got_include and got_library)


class build_ext_compile_args(build_ext):
    def build_extensions(self):
        compiler=self.compiler.compiler_type
        for ext in self.extensions:
            if compiler in extra_compile_args:
                ext.extra_compile_args=extra_compile_args[compiler]
                ext.extra_link_args=extra_link_args[compiler]
                ext.include_dirs.extend(list(extra_include_dirs[compiler]))
                ext.libraries.extend(list(extra_libraries[compiler]))

                opencl_include_and_libraries = check_for_opencl(compiler,ext.include_dirs,ext.library_dirs)
                if opencl_include_and_libraries:
                    ext.define_macros.append( ("CONVOLUTION_ENABLE_OPENCL",None) )
                    ext.libraries.append("OpenCL")
                    pass
                
                pass
            else:
                # use gcc parameters as default
                ext.extra_compile_args=extra_compile_args["gcc"]
                ext.extra_link_args=extra_link_args["gcc"]
                ext.include_dirs.extend(list(extra_include_dirs["gcc"]))
                ext.libraries.extend(extra_libraries["gcc"])
                pass
                
            pass
            
        
        build_ext.build_extensions(self)
        pass
    pass




class install_lib_save_version(install_lib):
    """Save version information"""
    def run(self):
        install_lib.run(self)
        
        for package in self.distribution.command_obj["build_py"].packages:
            install_dir=os.path.join(*([self.install_dir] + package.split('.')))
            fh=open(os.path.join(install_dir,"version.txt"),"w")
            fh.write("%s\n" % (version))  # version global, as created below
            fh.close()
            pass
        pass
    pass



# Extract GIT version
if os.path.exists(".git"):
    # Check if tree has been modified
    modified = subprocess.call(["git","diff-index","--quiet","HEAD","--"]) != 0
    
    gitrev = subprocess.check_output(["git","rev-parse","HEAD"]).strip()

    version = "git-%s" % (gitrev)

    # See if we can get a more meaningful description from "git describe"
    try:
        versionraw=subprocess.check_output(["git","describe","--tags","--match=v*"],stderr=subprocess.STDOUT).decode('utf-8').strip()
        # versionraw is like v0.1.0-50-g434343
        # for compatibility with PEP 440, change it to
        # something like 0.1.0+50.g434343
        matchobj=re.match(r"""v([^.]+[.][^.]+[.][^-.]+)(-.*)?""",versionraw)
        version=matchobj.group(1)
        if matchobj.group(2) is not None:
            version += '+'+matchobj.group(2)[1:].replace("-",".")
            pass
        pass
    except subprocess.CalledProcessError:
        # Ignore error, falling back to above version string
        pass

    if modified and version.find('+') >= 0:
        version += ".modified"
        pass
    elif modified:
        version += "+modified"
        pass
    pass
else:
    version = "UNKNOWN"
    pass

print("version = %s" % (version))

VibroSim_WelderModel_package_files = [ "pt_steps/*" ]

ext_modules=cythonize("VibroSim_WelderModel/convolution.pyx",language="c++)
#em_dict=dict([ (module.name,module) for module in ext_modules])
#conv_pyx_ext=em_dict["VibroSim_WelderModel.convolution"]
#conv_pyx_ext.include_dirs=["."]
##conv_pyx_ext.extra_compile_args=['-O0','-g']
#conv_pyx_ext.extra_compile_args=['-fopenmp','-O5']
#conv_pyx_ext.libraries=['gomp','OpenCL']



console_scripts=[ "vibrosim_plot_welder_motion" ]
console_scripts_entrypoints = [ "%s = VibroSim_WelderModel.bin.%s:main" % (script,script.replace("-","_")) for script in console_scripts ]



setup(name="VibroSim_WelderModel",
      description="Vibrothermography crack heating model simulation workflow",
      author="Stephen D. Holland",
      version=version,
      url="http://thermal.cnde.iastate.edu",
      zip_safe=False,
      ext_modules=ext_modules,
      packages=["VibroSim_WelderModel","VibroSim_WelderModel.bin"],
      cmdclass={"install_lib": install_lib_save_version,
                "build_ext": build_ext_compile_args,},
      package_data={"VibroSim_WelderModel": VibroSim_WelderModel_package_files},
      entry_points={ "limatix.processtrak.step_url_search_path": [ "limatix.share.pt_steps = VibroSim_WelderModel:getstepurlpath" ],
                     "console_scripts": console_scripts_entrypoints,
                 },
      python_requires='>=2.7.0')


