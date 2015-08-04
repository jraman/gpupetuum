#include <cstdlib>
#include <fstream>
#include <iostream>


void load_data(
    char * fea_file,
    char * label_file,
    int num_train_data,
    int num_units_ineach_layer
    ){

  float **input_features;
  int *output_labels;

  int feadim = num_units_ineach_layer;

  input_features=new float*[num_train_data];
  FILE *fp = fopen(fea_file,"rb");
  for(int i=0; i<num_train_data; i++){
    input_features[i]=new float[feadim];
    fread(input_features[i], sizeof(float), feadim, fp);
  }
  fclose(fp);

  output_labels=new int[num_train_data];
  std::ifstream infile;
  infile.open(label_file);
  for(int i=0;i<num_train_data;i++){
    infile>>output_labels[i];
  }
  infile.close();

  std::cout << "First feature vector:" << std::endl;
  float *tmpf = input_features[0];
  for (int ii=0; ii<10; ii++) {
    std::cout << *tmpf++ << ", ";
  }
  std::cout << "..." << std::endl;

  std::cout << "Last feature vector:" << std::endl;
  tmpf = input_features[num_train_data - 1];
  for (int ii=0; ii<10; ii++) {
    std::cout << *tmpf++ << ", ";
  }
  std::cout << "..." << std::endl;

  std::cout << "Labels:" << std::endl;
  int *tmpi = output_labels;
  for (int ii=0; ii<10; ii++) {
    std::cout << *tmpi++ << ", ";
  }
  std::cout << "..." << std::endl;
}


int main(int argc, char *argv[]) {
  // $0 ../imnet_data/imnet_0.bin ../imnet_data/0_label.txt 63336 21504
  const char usage[] = "USAGE: $0 data_filename, label_filename, num_samples, feature_vec_len";
  if (argc != 5) {
    std::cout << usage << std::endl;
    exit(1);
  }
  std::cout << "Reading " << argv[1] << ", " << argv[2] << std::endl;
  load_data(argv[1], argv[2], atoi(argv[3]), atoi(argv[4]));

  return 0;
}
