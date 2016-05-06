//
// Created by chenglong on 12/15/15.
//

#ifndef CAFFE_FILEUTIL_H
#define CAFFE_FILEUTIL_H

#include <vector>
#include <string>
#include <unistd.h>

int ListSubDir(const char *RootDir, std::vector<std::string> &subDirs);

int ListDirFile(const char *RootDir, const char *extension, std::vector<std::string> &files);

int CreateDir(const char *dirPath);

int RemoveDir(const char *dirPath);

std::string CreateNameByTime(const char *rootPath);

// TODO: inline function cannot linked???
bool inline dirExist(const char *path) { return access(path, 0) == 0; }

bool inline fileRead(const char *path) { return access(path, 0 | 4) == 0; }

// check whether file can be written
bool inline fileWrite(const char *path) { return access(path, 0 | 2) == 0; }

#endif //CAFFE_FILEUTIL_H
