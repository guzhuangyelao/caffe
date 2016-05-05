
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")

SET(LOCAL_LIB /home/chenglong/local/libzip-1.1.2/lib/libzip.so)
set(ZIP_INCLUDE_DIR /home/chenglong/local/libzip-1.1.2/include)
set(source_file ${PROJECT_SOURCE_DIR}/examples/12306/convert_saz_siamese_data.cpp)

set(util_files
  ${PROJECT_SOURCE_DIR}/examples/12306/util/jsoncpp.cpp
  ${PROJECT_SOURCE_DIR}/examples/12306/util/base64.cpp)

#foreach(source_file ${examples_srcs})
  # get file name
  get_filename_component(name ${source_file} NAME_WE)

  include_directories(
    ${ZIP_INCLUDE_DIR}
  )

  # get folder name
  get_filename_component(path ${source_file} PATH)
  get_filename_component(folder ${path} NAME_WE)
  add_executable(${name} ${source_file} ${util_files})
  target_link_libraries(${name} ${Caffe_LINK} ${LOCAL_LIB})
  caffe_default_properties(${name})

  # set back RUNTIME_OUTPUT_DIRECTORY
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/examples/${folder}")

  caffe_set_solution_folder(${name} examples)

  # install
  install(TARGETS ${name} DESTINATION bin)

  if(UNIX OR APPLE)
    # Funny command to make tutorials work
    # TODO: remove in future as soon as naming is standartaized everywhere
    set(__outname ${PROJECT_BINARY_DIR}/examples/${folder}/${name}${Caffe_POSTFIX})
    add_custom_command(TARGET ${name} POST_BUILD
                       COMMAND ln -sf "${__outname}" "${__outname}.bin")
  endif()
#endforeach()
