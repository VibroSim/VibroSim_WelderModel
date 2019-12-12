import sys
#from numpy.distutils.core import setup as numpy_setup, Extension as numpy_Extension
import os
import os.path
import subprocess
import re
import numpy as np
from setuptools import setup
from setuptools.command.install_lib import install_lib
from setuptools.command.install import install
from setuptools.command.build_ext import build_ext
import setuptools.command.bdist_egg
import sys
from Cython.Build import cythonize


extra_compile_args = {
    "msvc": ["/openmp"],
    #"unix": ["-O0", "-g", "-Wno-uninitialized"),    # Replace the line below with this line to enable debugging of the compiled extension
    "unix": ["-fopenmp","-O5","-Wno-uninitialized"],
    "clang": ["-fopenmp","-O5","-Wno-uninitialized"],
}

extra_include_dirs = {
    "msvc": [".",np.get_include()],
    "unix": [".",np.get_include()],
    "clang": [".",np.get_include()],
}

extra_libraries = {
    "msvc": [],
    "unix": ["gomp",],
    "clang": [],
}

extra_link_args = {
    "msvc": [],
    "unix": [],
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


    got_library =  compiler.find_library_file(library_dirs,"OpenCL")
    #if compiler=="msvc" or compiler=="mingw":
    #    for library_dir in library_dirs:
    #        if os.path.exists(os.path.join(library_dir,"OpenCL.lib")):
    #            got_library=True
    #            pass
    #        pass
    #
    #    pass
    #else: 
    #    for library_dir in library_dirs:
    #        if os.path.exists(os.path.join(library_dir,"libOpenCL.so")):
    #            got_library=True
    #            pass
    #        pass
    #    pass

    return (got_include,got_library)


class build_ext_compile_args(build_ext):
    def build_extensions(self):
        compiler=self.compiler.compiler_type

        if compiler=="unix":
            # Extract implicit include and library directories from compiler

            # If we need to test for gcc vs something else here,
            # do it based on substrings of self.compiler.compiler_so[0]
            
            # Run "echo | gcc -xc -e -v - -fsyntax_only" to get gcc config
            gccproc = subprocess.Popen(self.compiler.compiler_so + ["-xc","-E","-v","-","-fsyntax-only"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            (stdoutdata,stderrdata)=gccproc.communicate(input=b"")
            compiler_info = stderrdata.decode('utf-8').split("\n")
            # include directories are on lines between "#include <...> search starts here and "End of search list" and have a single space preceding: 
            includestartidx = compiler_info.index("#include <...> search starts here:")+1
            includeendidx = compiler_info.index("End of search list.")
            implicit_include_dirs = [ space_include_dir[1:] for space_include_dir in compiler_info[includestartidx:includeendidx] ]

            # Library directories are on a line that says LIBRARY_PATH=
            # and are separated by colons
            
            for stderrline in compiler_info:
                if stderrline.startswith("LIBRARY_PATH="):
                    library_path_str = stderrline[13:]
                    implicit_library_dirs = library_path_str.split(":")
                pass
            pass
        else:

            implicit_include_dirs = []
            implicit_library_dirs = []
            pass

        #import pdb
        #pdb.set_trace()
        
        for ext in self.extensions:
            if compiler in extra_compile_args:
                ext.extra_compile_args=extra_compile_args[compiler]
                ext.extra_link_args=extra_link_args[compiler]
                ext.include_dirs.extend(list(extra_include_dirs[compiler]))
                ext.libraries.extend(list(extra_libraries[compiler]))
                
                pass
            else:
                # use unix parameters as default
                ext.extra_compile_args=extra_compile_args["unix"]
                ext.extra_link_args=extra_link_args["unix"]
                ext.include_dirs.extend(list(extra_include_dirs["unix"]))
                ext.libraries.extend(extra_libraries["unix"])
                pass

            
            (got_opencl_include, got_opencl_library) = check_for_opencl(self.compiler,implicit_include_dirs + ext.include_dirs,implicit_library_dirs + ext.library_dirs)
            if got_opencl_include and got_opencl_library:
                print("OpenCL support (GPU acceleration) enabled.")
                
                
                ext.define_macros.append( ("CONVOLUTION_ENABLE_OPENCL",None) )
                ext.libraries.append("OpenCL")
                pass
            if not got_opencl_include:
                print("OpenCL include file CL/opencl.h not found in search path %s.\n" % (str(ext.include_dirs)))
                
                pass
            if not got_opencl_library:
                print("OpenCL library file (libOpenCL.so or OpenCL.lib) not found in search path %s.\n" % (str(ext.library_dirs)))
                
                pass
            if not (got_opencl_include and got_opencl_library):
                print("OpenCL support (GPU acceleration) disabled.")
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

ext_modules=cythonize("VibroSim_WelderModel/convolution.pyx",language="c++")
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


