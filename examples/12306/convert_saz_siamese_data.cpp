//
// Created by chenglong on 16-5-5.
//

#include <zip.h>
#include <string>
#include <glog/logging.h>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <leveldb/db.h>
#include <unordered_set>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/format.hpp"
#include "caffe/util/math_functions.hpp"

#include "util/json/json.h"
#include "util/base64.h"

const int ORIGIN_IMG_WIDTH = 293;
const int ORIGIN_IMG_HEIGHT = 190;
const int PIC_SIZE = 67;
const int TXT_WIDTH = 64;
const int TXT_HEIGHT = 25;

using namespace std;

vector<cv::Rect> ROI_RECT{
    cv::Rect(4, 41, PIC_SIZE, PIC_SIZE),
    cv::Rect(77, 41, PIC_SIZE, PIC_SIZE),
    cv::Rect(149, 41, PIC_SIZE, PIC_SIZE),
    cv::Rect(221, 41, PIC_SIZE, PIC_SIZE),
    cv::Rect(4, 113, PIC_SIZE, PIC_SIZE),
    cv::Rect(77, 113, PIC_SIZE, PIC_SIZE),
    cv::Rect(149, 113, PIC_SIZE, PIC_SIZE),
    cv::Rect(221, 113, PIC_SIZE, PIC_SIZE),
    cv::Rect(116, 0, TXT_WIDTH, TXT_HEIGHT)
};

int get_rect(int x, int y, cv::Rect &rect) {
  if (x < 5 || x > 293 || y < 5 || y > 190) {
    return -1;
  }
  int left, top;
  left = x < 74 ? 5 : x < 146 ? 77 : x < 218 ? 149 : 221;
  top = y < 110 ? 41 : 113;
  rect = cv::Rect(left, top, PIC_SIZE, PIC_SIZE);
  return 0;
}

int get_label(int x, int y) {
  int label = -1;
  /*
   * labels is given as bellow
   * 0 1 2 3
   * 4 5 6 7
   */
  if (x < 5 || x > 293 || y < 5 || y > 190) {
    return label;
  }
  int col = x < 74 ? 0 : x < 146 ? 1 : x < 218 ? 2 : 3;
  int row = y < 110 ? 0 : 1;
  label = row * 4 + col;
  return label;
}

int extract_image(zip_file_t *zip_f_p, string &img_buf) {
//  zip_file_t *zip_f_p = zip_fopen(z_p, test_arch_fn.c_str(), ZIP_FL_UNCHANGED);
  zip_uint64_t buf_size = 100 * 1024;
  if (zip_f_p == NULL) {
    return -1;
  }
  char *buf = new char[buf_size];
  zip_uint64_t len = zip_fread(zip_f_p, buf, buf_size);
  if (len < 0) {
    return -1;
  }
  vector<string> lines;
  boost::split(lines, buf, boost::is_any_of("\n"));
  delete[] buf;
  DLOG(INFO) << "lines number: " << lines.size() << " last line: " << lines.back();
  if (lines.size() == 0) {
    return -1;
  }

  Json::Value value;
  Json::Reader reader;
  if (!reader.parse(lines.back(), value)) {
    LOG(ERROR) << "parse data error!";
    return -1;
  } else {
    if (!value.isMember("img_buf")) {
      return -1;
    }
    string img_str = value["img_buf"].asCString();
    Base64Decode(img_str, &img_buf);
    DLOG(INFO) << "parse image buffer success, length: " << img_buf.length();
  }
  return 0;
}

int parse_label(const string &label_str, unordered_set<int> &labels) {
  vector<string> infos;
  if (label_str.length() == 0) {
    return -1;
  }
  boost::split(infos, label_str, boost::is_any_of(","));
  if (infos.size() % 2 != 0) {
    return -1;
  }
  for (unsigned int i = 0; i < infos.size()/2; i++) {
    auto x_s = infos[i];
    auto y_s = infos[i+1];
    boost::trim_if(x_s, boost::is_any_of("()"));
    boost::trim_if(y_s, boost::is_any_of("()"));
    int x = atoi(x_s.c_str());
    int y = atoi(y_s.c_str());
    int lbl = get_label(x, y);
    if (lbl >= 0) {
      labels.insert(lbl);
    }
  }
  return 0;
}

int extract_labels_info(zip_file_t *zip_f_p, unordered_set<int> &labels) {
  zip_uint64_t buf_size = 50 * 1024;
  if (zip_f_p == NULL) {
    return -1;
  }
  char *buf = new char[buf_size];
  zip_uint64_t len = zip_fread(zip_f_p, buf, buf_size);
  if (len < 0) {
    return -1;
  }
  vector<string> lines;
  boost::split(lines, buf, boost::is_any_of("\n"));
  delete[] buf;

  if (lines.size() == 0) {
    return -1;
  }

  Json::Value value;
  Json::Reader reader;
  if (!reader.parse(lines[lines.size() - 2],value)) {
    LOG(ERROR) << "parse data error! info: " << lines[lines.size()-2];
    return -1;
  } else {
    if (!value.isMember("res")) {
      return -1;
    }
    string label_str = value["res"].asCString();
    if (label_str.length() < 1) {
      return -1; // no result
    }
    LOG(INFO) << "label string: " << label_str;
    parse_label(label_str, labels);
    LOG(INFO) << "get labels size: " << labels.size();
  }
  return 0;
}

int load_image(const string &img_buf, cv::Mat &img) {
  cv::Mat data(1, img_buf.length(), CV_8UC1, (char*)img_buf.c_str());
  img = cv::imdecode(data, CV_LOAD_IMAGE_COLOR);
  return 0;
}

int process_zip() {
  return 0;
}

uint32_t swap_endian(uint32_t val) {
  val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
  return (val << 16) | (val >> 16);
}

void convert_dataset(const char* image_filename, const char* label_filename,
                     const char* db_filename) {
  // Open files
  std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
  std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
  CHECK(image_file) << "Unable to open file " << image_filename;
  CHECK(label_file) << "Unable to open file " << label_filename;
  // Read the magic and the meta data
  uint32_t magic;
  uint32_t num_items;
  uint32_t num_labels;
  uint32_t rows;
  uint32_t cols;

  image_file.read(reinterpret_cast<char*>(&magic), 4);
  magic = swap_endian(magic);
  CHECK_EQ(magic, 2051) << "Incorrect image file magic.";
  label_file.read(reinterpret_cast<char*>(&magic), 4);
  magic = swap_endian(magic);
  CHECK_EQ(magic, 2049) << "Incorrect label file magic.";
  image_file.read(reinterpret_cast<char*>(&num_items), 4);
  num_items = swap_endian(num_items);
  label_file.read(reinterpret_cast<char*>(&num_labels), 4);
  num_labels = swap_endian(num_labels);
  CHECK_EQ(num_items, num_labels);
  image_file.read(reinterpret_cast<char*>(&rows), 4);
  rows = swap_endian(rows);
  image_file.read(reinterpret_cast<char*>(&cols), 4);
  cols = swap_endian(cols);

  // Open leveldb
  leveldb::DB* db;
  leveldb::Options options;
  options.create_if_missing = true;
  options.error_if_exists = true;
  leveldb::Status status = leveldb::DB::Open(
      options, db_filename, &db);
  CHECK(status.ok()) << "Failed to open leveldb " << db_filename
      << ". Is it already existing?";

  char label_i;
  char label_j;
  char* pixels = new char[2 * rows * cols];
  std::string value;

  caffe::Datum datum;
  datum.set_channels(2);  // one channel for each image in the pair
  datum.set_height(rows);
  datum.set_width(cols);
  LOG(INFO) << "A total of " << num_items << " items.";
  LOG(INFO) << "Rows: " << rows << " Cols: " << cols;
  for (int itemid = 0; itemid < num_items; ++itemid) {
    int i = caffe::caffe_rng_rand() % num_items;  // pick a random  pair
    int j = caffe::caffe_rng_rand() % num_items;
    read_image(&image_file, &label_file, i, rows, cols,
               pixels, &label_i);
    read_image(&image_file, &label_file, j, rows, cols,
               pixels + (rows * cols), &label_j);
    datum.set_data(pixels, 2*rows*cols);
    if (label_i  == label_j) {
      datum.set_label(1);
    } else {
      datum.set_label(0);
    }
    datum.SerializeToString(&value);
    std::string key_str = caffe::format_int(itemid, 8);
    db->Put(leveldb::WriteOptions(), key_str, value);
  }

  delete db;
  delete [] pixels;
}

int main(int argc, char **argv) {

  google::InitGoogleLogging(argv[0]);
  FLAGS_logbufsecs = 0;
  google::SetStderrLogging(google::GLOG_INFO);

  if (argc < 2) {
    LOG(FATAL) << "useage: exe zip_fn";
    _exit(0);
  }
  string zip_fn = argv[1];

  int err_code;
  zip_t *z_p = zip_open(zip_fn.c_str(), ZIP_RDONLY, &err_code);
  if (z_p == NULL) {
    LOG(ERROR) << "open zip file: " << zip_fn << " error, code: " << err_code;
    return -1;
  }

  int file_num = zip_get_num_entries(z_p, 0);
  int img_num = (file_num - 2) / 3;
  LOG(INFO) << "find total " << file_num << " files in " << zip_fn;
  for (auto i = 1; i <= img_num; i++) {
    char img_arch_fn[30];
    char lbl_arch_fn[30];
    sprintf(img_arch_fn, "raw/%03d_c.txt", i);
    sprintf(lbl_arch_fn, "raw/%03d_s.txt", i);


    unordered_set<int> lbls;
    zip_file_t *lbl_f_p = zip_fopen(z_p, lbl_arch_fn, 0);
    if (extract_labels_info(lbl_f_p, lbls) < 0 || lbls.size() == 0) {
      continue;
    }

    zip_file_t *img_f_p = zip_fopen(z_p, img_arch_fn, 0);
    if (img_f_p != NULL) {
      string img_buf;
      if (extract_image(img_f_p, img_buf) < 0) {
        continue;
      }
      LOG(INFO) << "img_buf size: " << img_buf.length();
      cv::Mat img; //(ORIGIN_IMG_HEIGHT, ORIGIN_IMG_WIDTH, CV_8UC3, (void*)img_buf.c_str());
      load_image(img_buf, img);
      LOG(INFO) << "img size: (" << img.rows << ", " << img.cols << ")";
      cv::imwrite("test1.jpg", img);

      for(unsigned int lbl_idx = 0; lbl_idx < 8; lbl_idx++) {
        cv::Rect &roi = ROI_RECT[lbl_idx];
        LOG(INFO) << roi.height << " " << roi.width;
        cv::Mat sub_img = img(roi);
        char sub_img_fn[100];
        sprintf(sub_img_fn, "%03d_%d.jpg", i, lbl_idx);
        cv::imwrite(sub_img_fn, sub_img);
        LOG(INFO) << "sub image size: (" << sub_img.rows << ", " << sub_img.cols << ")";

      }
    } else {
      LOG(ERROR) << "open zip archive file: " << img_arch_fn << " error!";
    }
    break;

  }


/*
string test_arch_fn = "raw/002_c.txt";
int ret = zip_name_locate(z_p, test_arch_fn.c_str(), ZIP_FL_ENC_GUESS);
if (ret != 1) {
  return -1;
}
zip_file_t *zip_f_p = zip_fopen(z_p, test_arch_fn.c_str(), ZIP_FL_UNCHANGED);

string img_buf;
if (extract_image(zip_f_p, img_buf) < 0) {
  return -1;
}

cv::Mat img = cv::Mat(190, 293, CV_8UC3, (void *)img_buf.c_str());
//  ofstream os("test.jpg", ios::binary);
//  os.write(img_buf.c_str(), img_buf.length());
//  os.close();
LOG(INFO) << "image rows: " << img.rows << " cols: " << img.cols;

string test_lbl_fn = "raw/001_s.txt";
ret = zip_name_locate(z_p, test_lbl_fn.c_str(), ZIP_FL_ENC_GUESS);
if (ret != 1) {
  LOG(ERROR) << "archive file: " << test_lbl_fn << " not in zip file: " << zip_fn;
  return -1;
}
zip_f_p = zip_fopen(z_p, test_lbl_fn.c_str(), ZIP_FL_UNCHANGED);
string lbl_str;
unordered_set<int> labels;
if (extract_labels_info(zip_f_p, labels) < 0) {
  LOG(ERROR) << "extract labels info error!";
  return -1;
}
   */

  return 0;
}
