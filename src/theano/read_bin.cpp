
void dnn::load_data(char * fea_file, char * label_file){
  int feadim=num_units_ineach_layer[0];

  input_features=new float*[num_train_data];
  FILE *fp=fopen(fea_file,"rb");
  for(int i=0;i<num_train_data;i++){
    input_features[i]=new float[feadim];
    fread(input_features[i], sizeof(float),feadim, fp);
  }
  fclose(fp);
  output_labels=new int[num_train_data];
  std::ifstream infile;
  infile.open(label_file);
  for(int i=0;i<num_train_data;i++){
    infile>>output_labels[i];
  }
  infile.close();
}
