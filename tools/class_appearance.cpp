// 
// cmd:
// class_appearance net_proto pretrained_net_proto mean outpath iteration step label [input_image/binary]
// ../../build/tools/class_appearance.bin imagenet_saliency_nodrop.prototxt 89caffe_imagenet_train_iter_21000 8000 500 4 

#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/proto/caffe.pb.h"

#define VIRTUAL_LABEL 0;
#define WIDTH 227
#define HEIGHT 227
#define CHANNEL 3
#define BATCHSIZE 1
#define NMAX 10000
#define NMIN (-10000)
#define NAMELEN 100
#define LAMBDA 0.05
#define INFOITER 50
#define DRAWITER 500
#define DELTA 100
#define NEGMIN (-0.0)
#define POSMIN 0.0
#define MINDIFF (-1)
#define MOMENTUM 0.9
#define WEIGHTDECAY 0.0005

using namespace caffe;  // NOLINT(build/namespaces)


int ReadImageToBlob(const string& filename, const int label,
                    const int height, const int width, Blob<float>& image_blob)
{
    Datum datum;
    bool ret = ReadImageToDatum(filename, label, height, width, &datum);
    if (!ret) {
        LOG(ERROR) << "read " << filename << " failed";
        return -1;
    }
    float* dto= image_blob.mutable_cpu_data();
    float* d_diff = image_blob.mutable_cpu_diff();
    const char* dfrom = datum.data().c_str();    
    
    for (int n_id = 0; n_id<1; n_id++){
        for (int c = 0; c < CHANNEL; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    *dto = static_cast<float> (*dfrom);
                    *d_diff++ = 0;
                    dto++;
                    ++dfrom;
                }
            }
        }
    }
    return 1;
}

void InitDataBlob( Blob<float>& image_blob, const char* image = NULL )
{
  float* dto = image_blob.mutable_cpu_data();  
  float* d_diff = image_blob.mutable_cpu_diff();
  for (int i = 0; i < image_blob.count(); ++i)
  {
    *dto++ = rand()%DELTA - 0.5 * DELTA; // INIT random
    *d_diff++ = 0; // history blob --momentum only use the diff data
  }
}

int BlobCopyDiff( Blob<float>& to_blob, Blob<float>& from_blob )
{
  int copy_num = to_blob.count();
  if( copy_num != from_blob.count() )
    return 0;
  float* to_diff = to_blob.mutable_cpu_diff();
  const float* from_diff = from_blob.cpu_diff();
  for (int i = 0; i < copy_num; ++i)
  {
    to_diff[i] = from_diff[i];
  }
  return 1;
}

template <typename Dtype>
Dtype CalcL1Norm(const Dtype* d_from, int d_count) {
  
  double sum = 0.0;
  for (int i = 0; i < d_count; ++i) {
    sum += fabs(d_from[i]);
  }
  return sum;
}

template <typename Dtype>
bool DataMaxMin( const Dtype* d_from, int d_count, float& d_max, float& d_min )
{
  d_max = NMIN;
  d_min = NMAX;
  for (int i = 0; i < d_count; ++i)
  {
    if(d_max<d_from[i]){
      d_max = d_from[i];
    }
    if(d_min>d_from[i]){
      d_min = d_from[i];
    }
  }
  return 1;
}

template <typename Dtype>
bool DataMaxMin( const Dtype* d_from, int d_count, float& d_max, float& d_min, int& max_pos, int& min_pos )
{
  d_max = NMIN;
  d_min = NMAX;
  max_pos = 0;
  min_pos = 0;
  for (int i = 0; i < d_count; ++i)
  {
    if(d_max<d_from[i]){
      d_max = d_from[i];
      max_pos = i;
    }
    if(d_min>d_from[i]){
      d_min = d_from[i];
      min_pos = i;
    }
  }
  return 1;
}

bool UpdateBlob( Blob<float>& blob, float learn_rate, Blob<float>& last_blob)
{
  float* mut_data = blob.mutable_cpu_data();
  const float* diff = blob.cpu_diff();
  float* last_diff = last_blob.mutable_cpu_diff();
  const float* data = blob.cpu_data();
  float d_max;
  float d_min;
  float delta_tmp;
  
  for (int i = 0; i < blob.count(); ++i)
  {
    delta_tmp = MOMENTUM * last_diff[i] - ( WEIGHTDECAY * mut_data[i] + diff[i] ) * learn_rate;
    mut_data[i] += delta_tmp;
    last_diff[i] = delta_tmp;
  }
  DataMaxMin<float>(  data, blob.count(), d_max, d_min );
  for (int i = 0; i < blob.count(); ++i)
  {
    mut_data[i] = DELTA*( (mut_data[i] - d_min)/(d_max - d_min) - 0.5 );
  }
  return 1;
}

void blob2image(Blob<float>& image_blob, char* image_name){
    const float* dfrom= image_blob.cpu_data();
    int width = image_blob.width();
    int height = image_blob.height();
    int img_size = width*height;
    int channels = image_blob.channels();
    cv::Mat tmp_img(width,height, CV_32FC3);
    cv::Mat tmp_gray; //(width,height, CV_8UC1);
    char temp_name[NAMELEN];

    float d_max;
    float d_min;
    DataMaxMin( dfrom, image_blob.count(), d_max, d_min );
    LOG(ERROR) << "blob min: " << d_min << "  max: " << d_max;
    float alpha = 255./(d_max - d_min);

    for (int img_id=0; img_id<image_blob.num(); img_id++){//image_blob.nums()
        sprintf(temp_name, "%s_%d.jpg", image_name, img_id);
        for (int c = 0; c < channels; ++c)
        {
            
            for(int h=0; h<height; h++){
                for(int w=0; w<width; w++){
                    // int p_id = img_id*3*img_size + c*img_size + h*width + w;
                    // tmp_img.at<cv::Vec3f>(w, h)[0] = dfrom[img_id*3*img_size + c*img_size + h*width + w];
                    // tmp_img.at<cv::Vec3f>(w, h)[1] = *(dfrom+size);
                    // tmp_img.at<cv::Vec3f>(w, h)[2] = *(dfrom+2*size);

                    tmp_img.at<cv::Vec3f>(h, w)[2-c] = alpha * ( (*dfrom) - d_min);
                    dfrom++;
                }
            }
            
        }

        // tmp_img.convertTo(tmp_gray, CV_8UC1);
        cv::imwrite(temp_name, tmp_img);
    }
}

void blob2image(Blob<float>* image_blob, char* image_name){
    blob2image(*image_blob, image_name);
}

void blob2image(Blob<float>& image_blob, char* image_name, Blob<float>& mean_blob){
    const float* dfrom= image_blob.cpu_data();
    const float* dmean= mean_blob.cpu_data();
    
    int width = image_blob.width();
    int height = image_blob.height();
    int img_size = width*height;
    int channels = image_blob.channels();
    //printf("channels: %d\n", channels);
    cv::Mat tmp_img(width,height, CV_32FC3);
    cv::Mat tmp_gray; //(width,height, CV_8UC1);
    char temp_name[NAMELEN];

    int h_off = 0;
    int w_off = 0; //暂时没有便宜
    
    int num_tmp;
    for (int img_id=0; img_id<image_blob.num(); img_id++){//image_blob.nums()
        sprintf(temp_name, "%s_%d.jpg", image_name, img_id);
        num_tmp = img_id * channels * height * width;
        int c_tmp;
        int c_tmp_mean;
        for (int c = 0; c < channels; ++c)
        {
            c_tmp_mean = c * height * width;
            c_tmp = num_tmp + c_tmp_mean;
            int h_tmp_mean;
            int h_tmp;
            for(int h=0; h<height; h++){
                h_tmp_mean = c_tmp_mean + (h + h_off) * width;
                h_tmp = c_tmp + h * width;
                for(int w=0; w<width; w++){
                    tmp_img.at<cv::Vec3f>(h, w)[c] = dfrom[h_tmp + w] + dmean[h_tmp_mean + w + w_off];                    
                }
            }
            
        }
        cv::imwrite(temp_name, tmp_img);
    }
}

void Blob2BinFile( Blob<float>* image_blob, const char* binfile_head )
{
  BlobProto write_proto;
  char temp_name[NAMELEN];
  sprintf(temp_name, "%s.binaryproto", binfile_head);
  image_blob->ToProto( &write_proto, true);
  WriteProtoToBinaryFile( write_proto, temp_name );
}

/*
功能： 打印blob vec的相关信息
*/
int info(Blob<float>& blob)
{
  printf("blob num: %d  channel: %d  width: %d  height: %d\n", 
             blob.num(), blob.channels(), blob.width(), blob.height() );
  printf("L1 of blob data: %f\n", CalcL1Norm<float> ( blob.cpu_data(), blob.count() ) );
  printf("L1 of blob diff: %f\n", CalcL1Norm<float> ( blob.cpu_diff(), blob.count() ) );  
}

int info(const vector< Blob<float>* >& blob_vec)
{
  printf("*****************************\n");
  int vec_size = blob_vec.size();
  printf("blob_vec size: %d\n", vec_size );
  for (int i = 0; i < vec_size; ++i)
  {
    printf("blob id: %d ", i);
    Blob<float>* cur_blob = blob_vec[i];
    info( *blob_vec[i] );
  }
  return 1;
}

/*
功能： 打印blob vec的相关信息
*/
int info(const vector< shared_ptr<Blob<float> > >& blob_vec)
{
  printf("*****************************\n");
  int vec_size = blob_vec.size();
  printf("blob_vec size: %d\n", vec_size );
  for (int i = 0; i < vec_size; ++i)
  {
    printf("blob id: %d ", i);
    shared_ptr<Blob<float> > cur_blob = blob_vec[i];
    info( *blob_vec[i] );
  }
  return 1;
}



int main(int argc, char** argv) {
  if (argc < 7 || argc > 9) {
    LOG(ERROR) << "class_appearance net_proto pretrained_net_proto mean outpath iteration step label [input_image]";
    return 1;
  }

  Caffe::set_mode(Caffe::GPU);
  Caffe::SetDevice(0);

  Net<float> caffe_deconv_net(argv[1], TRAIN);
  caffe_deconv_net.CopyTrainedLayersFrom(argv[2]);

  // load mean for draw the class image 
  LOG(ERROR) << "load mean file ok from: " << argv[3];
  BlobProto input_mean;
  ReadProtoFromBinaryFile(argv[3], &input_mean);
  Blob<float> mean_blob;
  mean_blob.FromProto(input_mean);
  LOG(ERROR) << "Done...";
  
  int iteration = atoi(argv[5]);
  float learn_rate = atof(argv[6]);

  vector<Blob<float>*> image_blob_input_vec;
  Blob<float> image_blob(BATCHSIZE, CHANNEL, HEIGHT, WIDTH); // input 
  Blob<float> history_blob(BATCHSIZE, CHANNEL, HEIGHT, WIDTH); // momentum
  InitDataBlob( history_blob );
  char tmp_out[NAMELEN];

  
  // go on all labels or only one
  // for (int true_label = 0; true_label < 5; ++true_label)
  // {
  //   // only one label
  //   if (true_label != atoi(argv[6]) ) {
  //     continue;
  //   }

    // init bottom input
  bool load_binary = false;
    int true_label = atoi(argv[7]);
    if( argc > 8 )
    {
      // const char *p = strrchr( argv[8], '.');
      char *p = strrchr(argv[8], '.');
      LOG(ERROR) << "*************load image from: " << argv[8] << "*************";
      if ( strcmp(p, ".jpg") == 0 )
      {
        // load image
        if( ReadImageToBlob( argv[8], 1, HEIGHT, WIDTH, image_blob) ){
          LOG(ERROR) << "Done!";
          UpdateBlob( image_blob, learn_rate, history_blob );
        }
        else
        LOG(ERROR) << "Failed!!!";
      }
      else
      {
        // load binary file, use the input history as temp proto
        ReadProtoFromBinaryFile(argv[8], &input_mean);
        image_blob.FromProto(input_mean);
        LOG(ERROR) << "Done!";
        LOG(ERROR) << "*************load diff history from: " << argv[8] << "*************";
        assert( history_blob.count() == image_blob.count() );
        BlobCopyDiff( history_blob, image_blob );
        load_binary = true;
        LOG(ERROR) << "Done!";
      }
      
    }
    else{
      LOG(ERROR) << "*************Init input image as Random/Zero*************";
      InitDataBlob( image_blob );
    }
    
    image_blob_input_vec.clear();
    image_blob_input_vec.push_back( &image_blob );
    caffe_deconv_net.bottom_vecs()[1][0]->CopyFrom(image_blob);

    // printf("++++++++++++++++++++++++++++ before forward ++++++++++++++++++++++++++++\n");
    // info( caffe_deconv_net.blobs() );

    float last_delta = MINDIFF;
    // iter to update the class image
    LOG(ERROR) << "******************Iteration:**" << iteration << "**begin!***********************";
    for (int iter = 0; iter < iteration; iter++) {
      // forward
      float loss = 0;
      for (int i = 1; i < caffe_deconv_net.layers().size(); ++i) {
        // LOG(ERROR) << "Forwarding " << layer_names_[i];
        float layer_loss = caffe_deconv_net.layers()[i]->Forward(caffe_deconv_net.bottom_vecs()[i], caffe_deconv_net.top_vecs()[i]);
        loss += layer_loss;
      }

      // printf("++++++++++++++++++++++++++++ %d forward ++++++++++++++++++++++++++++\n", iter+1);
      // info( caffe_deconv_net.blobs() );

      const vector<Blob<float>*>& output = caffe_deconv_net.output_blobs();
      // printf("outblob:   xxxxxxxx\n");
      // info( output );
      
      Blob<float>* activation = output[0];

      assert(activation->num() == 1 && activation->height() == 1 && activation->width() == 1);
      
      bool draw_flag = true; // flag to draw class image

      // get the L2 of I to norm 
      // float norm = CalcL1Norm<float>( caffe_deconv_net.blobs()[0]->cpu_data(),
      //                                caffe_deconv_net.blobs()[0]->count() );

      // get the diff of the last layer
      if ( iter%INFOITER == 0 ){
              printf( "iter %d  ", iter);
      }
      float* diff = activation->mutable_cpu_diff();
      for (int n = 0; n < activation->num(); n++) {
        float right_psb = activation->data_at(n, true_label, 0, 0);
        
        for (int c = 0; c < activation->channels(); c++) {
          int offset = activation->offset(n, c, 0, 0);
          // if ( iter%INFOITER == 0 ){
          // //     printf( "  out%d: %f", c, activation->data_at(n, c, 0, 0) );
          // }
          if (c == true_label) {
            if ( iter%INFOITER == 0 ){
              printf( "  output%d: %f", c, activation->data_at(n, c, 0, 0) );
            }
            if( false ) { //load_binary && iter == 0 ) {
              // init the diff from the load blob
              diff[offset] = -image_blob.cpu_diff()[0];
            }
            else {              
              diff[offset] = -activation->data_at(n, c, 0, 0); //- last_delta; // + LAMBDA * norm 
              if ( diff[offset] > 0 )
              {
                diff[offset] = -0.2;
              }
            }
            //last_delta = activation->data_at(n, c, 0, 0);
            // if the output is a little negative, then give back a little possitive 
            // to pass the zero, now it is of no use( the condition bellow is always false ), just have a try to pass the zero
            if (diff[offset] > NEGMIN && diff[offset] < 0)
            {
              diff[offset] = POSMIN;
            }
            
          }
          else {
            draw_flag = draw_flag && (right_psb > activation->data_at(n, c, 0, 0));
            diff[offset] = 0;
          }          
        }
      }

      // backward
      caffe_deconv_net.Backward();

      float max_recons;
      float min_recons;
      int max_pos;
      int min_pos;
      
      // update the input image
      shared_ptr<Blob<float> > recons = caffe_deconv_net.blobs()[0];
      UpdateBlob( *recons, learn_rate, history_blob );
      DataMaxMin(  recons->cpu_diff(), recons->count(), max_recons, min_recons, max_pos, min_pos );

      // for image draw at end
      // image_blob.CopyFrom(*caffe_deconv_net.bottom_vecs()[1][0]);
      // history_blob.CopyFrom( *caffe_deconv_net.bottom_vecs()[1][0], 1 );
      // info and plot image, save binary file
      if ( iter % INFOITER == 0 )
        printf( "     input_dif max: %f min: %f  (%d, %d)  draw: %d\n", max_recons, min_recons, max_pos, min_pos, draw_flag );
      if (iter % DRAWITER == 0 && draw_flag)
      {
        Blob<float>* save_blob = caffe_deconv_net.bottom_vecs()[1][0];
        sprintf(tmp_out, "%s/class_%d_%d", argv[4], true_label, iter);
        blob2image(*save_blob, tmp_out); // , mean_blob
        assert( save_blob->count() == activation->count() );

        Blob2BinFile( caffe_deconv_net.bottom_vecs()[1][0], tmp_out );
      }
    }
    // sprintf(tmp_out, "class_%d", true_label);
    // blob2image(image_blob, tmp_out, mean_blob);
  // }

  return 0;
}