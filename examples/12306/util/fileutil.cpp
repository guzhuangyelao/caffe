//
// Created by chenglong on 12/15/15.
//

#include <dirent.h>
#include <string.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "fileutil.h"

using namespace std;

int ListSubDir(const char *RootDir, vector<string> &subDirs) {
  subDirs.clear();
  struct dirent *ent = NULL;
  DIR *pDir;
  struct stat ent_stat;
  pDir = opendir(RootDir);
  string subDirName;
  while ((ent = readdir(pDir)) != NULL) {
    if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
      continue;

    subDirName = string(RootDir) + '/' + ent->d_name;
    stat(subDirName.c_str(), &ent_stat);
    if (S_ISDIR(ent_stat.st_mode)) {
      subDirs.push_back(subDirName);
    }
  }
  closedir(pDir);
  return 0;
}

int ListDirFile(const char *RootDir, const char *extension, std::vector<std::string> &files) {
  files.clear();
  struct dirent *ent = NULL;
  DIR *pDir;
  struct stat ent_stat;
  pDir = opendir(RootDir);
  string filename;
  while ((ent = readdir(pDir)) != NULL) {
    if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
      continue;

    filename = string(RootDir) + '/' + ent->d_name;
    stat(filename.c_str(), &ent_stat);
    if (S_ISREG(ent_stat.st_mode)) {
      const char *p = strrchr(filename.c_str(), '.');
      if (p != NULL && (strcmp(extension, p + 1) == 0)) {
        files.push_back(filename);
      }
    }
  }
  closedir(pDir);
  return 0;
}

int CreateDir(const char *dirPath) {
  char command[1024];
  sprintf(command, "mkdir %s", dirPath);
  if (system(command) == 0) {
    return 0;
  }
  return -1;
}

int RemoveDir(const char *dirPath) {
  char command[1024];
  sprintf(command, "rm -rf %s", dirPath);
  if (system(command) == 0) {
    return 0;
  }
  return -1;
}

std::string CreateNameByTime(const char *rootPath) {
  char name[1024];
  timeval tv;
  gettimeofday(&tv, NULL);
  sprintf(name, "%s/%ld_%ld/", rootPath, tv.tv_sec, tv.tv_usec);
  return name;
}
