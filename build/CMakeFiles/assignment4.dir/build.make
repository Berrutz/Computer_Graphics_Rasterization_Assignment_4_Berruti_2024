# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/xxx/Desktop/CG/2024/Assignment4/Assignment_4_Berruti_2024

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/xxx/Desktop/CG/2024/Assignment4/Assignment_4_Berruti_2024/build

# Include any dependencies generated for this target.
include CMakeFiles/assignment4.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/assignment4.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/assignment4.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/assignment4.dir/flags.make

CMakeFiles/assignment4.dir/src/ex2/ex2_main.cpp.o: CMakeFiles/assignment4.dir/flags.make
CMakeFiles/assignment4.dir/src/ex2/ex2_main.cpp.o: /home/xxx/Desktop/CG/2024/Assignment4/Assignment_4_Berruti_2024/src/ex2/ex2_main.cpp
CMakeFiles/assignment4.dir/src/ex2/ex2_main.cpp.o: CMakeFiles/assignment4.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/xxx/Desktop/CG/2024/Assignment4/Assignment_4_Berruti_2024/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/assignment4.dir/src/ex2/ex2_main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/assignment4.dir/src/ex2/ex2_main.cpp.o -MF CMakeFiles/assignment4.dir/src/ex2/ex2_main.cpp.o.d -o CMakeFiles/assignment4.dir/src/ex2/ex2_main.cpp.o -c /home/xxx/Desktop/CG/2024/Assignment4/Assignment_4_Berruti_2024/src/ex2/ex2_main.cpp

CMakeFiles/assignment4.dir/src/ex2/ex2_main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/assignment4.dir/src/ex2/ex2_main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xxx/Desktop/CG/2024/Assignment4/Assignment_4_Berruti_2024/src/ex2/ex2_main.cpp > CMakeFiles/assignment4.dir/src/ex2/ex2_main.cpp.i

CMakeFiles/assignment4.dir/src/ex2/ex2_main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/assignment4.dir/src/ex2/ex2_main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xxx/Desktop/CG/2024/Assignment4/Assignment_4_Berruti_2024/src/ex2/ex2_main.cpp -o CMakeFiles/assignment4.dir/src/ex2/ex2_main.cpp.s

CMakeFiles/assignment4.dir/src/ex2/ex2_raster.cpp.o: CMakeFiles/assignment4.dir/flags.make
CMakeFiles/assignment4.dir/src/ex2/ex2_raster.cpp.o: /home/xxx/Desktop/CG/2024/Assignment4/Assignment_4_Berruti_2024/src/ex2/ex2_raster.cpp
CMakeFiles/assignment4.dir/src/ex2/ex2_raster.cpp.o: CMakeFiles/assignment4.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/xxx/Desktop/CG/2024/Assignment4/Assignment_4_Berruti_2024/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/assignment4.dir/src/ex2/ex2_raster.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/assignment4.dir/src/ex2/ex2_raster.cpp.o -MF CMakeFiles/assignment4.dir/src/ex2/ex2_raster.cpp.o.d -o CMakeFiles/assignment4.dir/src/ex2/ex2_raster.cpp.o -c /home/xxx/Desktop/CG/2024/Assignment4/Assignment_4_Berruti_2024/src/ex2/ex2_raster.cpp

CMakeFiles/assignment4.dir/src/ex2/ex2_raster.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/assignment4.dir/src/ex2/ex2_raster.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xxx/Desktop/CG/2024/Assignment4/Assignment_4_Berruti_2024/src/ex2/ex2_raster.cpp > CMakeFiles/assignment4.dir/src/ex2/ex2_raster.cpp.i

CMakeFiles/assignment4.dir/src/ex2/ex2_raster.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/assignment4.dir/src/ex2/ex2_raster.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xxx/Desktop/CG/2024/Assignment4/Assignment_4_Berruti_2024/src/ex2/ex2_raster.cpp -o CMakeFiles/assignment4.dir/src/ex2/ex2_raster.cpp.s

# Object files for target assignment4
assignment4_OBJECTS = \
"CMakeFiles/assignment4.dir/src/ex2/ex2_main.cpp.o" \
"CMakeFiles/assignment4.dir/src/ex2/ex2_raster.cpp.o"

# External object files for target assignment4
assignment4_EXTERNAL_OBJECTS =

assignment4: CMakeFiles/assignment4.dir/src/ex2/ex2_main.cpp.o
assignment4: CMakeFiles/assignment4.dir/src/ex2/ex2_raster.cpp.o
assignment4: CMakeFiles/assignment4.dir/build.make
assignment4: CMakeFiles/assignment4.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/xxx/Desktop/CG/2024/Assignment4/Assignment_4_Berruti_2024/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable assignment4"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/assignment4.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/assignment4.dir/build: assignment4
.PHONY : CMakeFiles/assignment4.dir/build

CMakeFiles/assignment4.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/assignment4.dir/cmake_clean.cmake
.PHONY : CMakeFiles/assignment4.dir/clean

CMakeFiles/assignment4.dir/depend:
	cd /home/xxx/Desktop/CG/2024/Assignment4/Assignment_4_Berruti_2024/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xxx/Desktop/CG/2024/Assignment4/Assignment_4_Berruti_2024 /home/xxx/Desktop/CG/2024/Assignment4/Assignment_4_Berruti_2024 /home/xxx/Desktop/CG/2024/Assignment4/Assignment_4_Berruti_2024/build /home/xxx/Desktop/CG/2024/Assignment4/Assignment_4_Berruti_2024/build /home/xxx/Desktop/CG/2024/Assignment4/Assignment_4_Berruti_2024/build/CMakeFiles/assignment4.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/assignment4.dir/depend

