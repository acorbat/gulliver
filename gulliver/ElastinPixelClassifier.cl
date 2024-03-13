/*
OpenCL RandomForestClassifier
classifier_class_name = PixelClassifier
feature_specification = gaussian_blur=1 difference_of_gaussian=1 laplace_box_of_gaussian_blur=1 sobel_of_gaussian_blur=1 gaussian_blur=2 gaussian_blur=3 gaussian_blur=4 gaussian_blur=5 difference_of_gaussian=5 difference_of_gaussian=4 difference_of_gaussian=3 difference_of_gaussian=2 laplace_box_of_gaussian_blur=2 laplace_box_of_gaussian_blur=3 sobel_of_gaussian_blur=3 sobel_of_gaussian_blur=2 sobel_of_gaussian_blur=4 laplace_box_of_gaussian_blur=4 laplace_box_of_gaussian_blur=5 sobel_of_gaussian_blur=5 gaussian_blur=10 difference_of_gaussian=10 gaussian_blur=15 gaussian_blur=25 original
num_ground_truth_dimensions = 2
num_classes = 2
num_features = 25
max_depth = 3
num_trees = 100
feature_importances = 0.1056542778654233,0.006000395946152028,0.0018413013054184866,0.0013166895824389504,0.09770145444902924,0.010161185880802379,0.0012122323105110998,0.0016627150862243963,0.018260026596774077,0.06671048960641361,0.10684898892782897,0.01851430167828193,0.026818833796285097,0.15238959186943496,0.0005376750113338617,0.0008393949998432615,0.00040490207226493075,0.094557165647211,0.02774445812929568,0.0013858106903460684,0.0016277283201263475,0.009526032191158782,0.0004793800360546388,0.0009415933805460374,0.24686337462080088
apoc_version = 0.12.0
*/
__kernel void predict (IMAGE_in0_TYPE in0, IMAGE_in1_TYPE in1, IMAGE_in2_TYPE in2, IMAGE_in3_TYPE in3, IMAGE_in4_TYPE in4, IMAGE_in5_TYPE in5, IMAGE_in6_TYPE in6, IMAGE_in7_TYPE in7, IMAGE_in8_TYPE in8, IMAGE_in9_TYPE in9, IMAGE_in10_TYPE in10, IMAGE_in11_TYPE in11, IMAGE_in12_TYPE in12, IMAGE_in13_TYPE in13, IMAGE_in14_TYPE in14, IMAGE_in15_TYPE in15, IMAGE_in16_TYPE in16, IMAGE_in17_TYPE in17, IMAGE_in18_TYPE in18, IMAGE_in19_TYPE in19, IMAGE_in20_TYPE in20, IMAGE_in21_TYPE in21, IMAGE_in22_TYPE in22, IMAGE_in23_TYPE in23, IMAGE_in24_TYPE in24, IMAGE_out_TYPE out) {
 sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
 const int x = get_global_id(0);
 const int y = get_global_id(1);
 const int z = get_global_id(2);
 float i0 = READ_IMAGE(in0, sampler, POS_in0_INSTANCE(x,y,z,0)).x;
 float i1 = READ_IMAGE(in1, sampler, POS_in1_INSTANCE(x,y,z,0)).x;
 float i2 = READ_IMAGE(in2, sampler, POS_in2_INSTANCE(x,y,z,0)).x;
 float i3 = READ_IMAGE(in3, sampler, POS_in3_INSTANCE(x,y,z,0)).x;
 float i4 = READ_IMAGE(in4, sampler, POS_in4_INSTANCE(x,y,z,0)).x;
 float i5 = READ_IMAGE(in5, sampler, POS_in5_INSTANCE(x,y,z,0)).x;
 float i6 = READ_IMAGE(in6, sampler, POS_in6_INSTANCE(x,y,z,0)).x;
 float i7 = READ_IMAGE(in7, sampler, POS_in7_INSTANCE(x,y,z,0)).x;
 float i8 = READ_IMAGE(in8, sampler, POS_in8_INSTANCE(x,y,z,0)).x;
 float i9 = READ_IMAGE(in9, sampler, POS_in9_INSTANCE(x,y,z,0)).x;
 float i10 = READ_IMAGE(in10, sampler, POS_in10_INSTANCE(x,y,z,0)).x;
 float i11 = READ_IMAGE(in11, sampler, POS_in11_INSTANCE(x,y,z,0)).x;
 float i12 = READ_IMAGE(in12, sampler, POS_in12_INSTANCE(x,y,z,0)).x;
 float i13 = READ_IMAGE(in13, sampler, POS_in13_INSTANCE(x,y,z,0)).x;
 float i14 = READ_IMAGE(in14, sampler, POS_in14_INSTANCE(x,y,z,0)).x;
 float i15 = READ_IMAGE(in15, sampler, POS_in15_INSTANCE(x,y,z,0)).x;
 float i16 = READ_IMAGE(in16, sampler, POS_in16_INSTANCE(x,y,z,0)).x;
 float i17 = READ_IMAGE(in17, sampler, POS_in17_INSTANCE(x,y,z,0)).x;
 float i18 = READ_IMAGE(in18, sampler, POS_in18_INSTANCE(x,y,z,0)).x;
 float i19 = READ_IMAGE(in19, sampler, POS_in19_INSTANCE(x,y,z,0)).x;
 float i20 = READ_IMAGE(in20, sampler, POS_in20_INSTANCE(x,y,z,0)).x;
 float i21 = READ_IMAGE(in21, sampler, POS_in21_INSTANCE(x,y,z,0)).x;
 float i22 = READ_IMAGE(in22, sampler, POS_in22_INSTANCE(x,y,z,0)).x;
 float i23 = READ_IMAGE(in23, sampler, POS_in23_INSTANCE(x,y,z,0)).x;
 float i24 = READ_IMAGE(in24, sampler, POS_in24_INSTANCE(x,y,z,0)).x;
 float s0=0;
 float s1=0;
if(i24<14606.0){
 if(i13<478.27587890625){
  s0+=1164.0;
 } else {
  if(i1<-192.890625){
   s0+=7.0;
  } else {
   s1+=1.0;
  }
 }
} else {
 if(i4<20633.751953125){
  if(i19<9637.1552734375){
   s0+=1.0;
   s1+=42.0;
  } else {
   s0+=3.0;
   s1+=1.0;
  }
 } else {
  if(i0<21801.052734375){
   s0+=1.0;
   s1+=6.0;
  } else {
   s1+=727.0;
  }
 }
}
if(i13<601.5966796875){
 if(i7<26077.337890625){
  if(i7<20148.97265625){
   s0+=1142.0;
   s1+=2.0;
  } else {
   s0+=42.0;
   s1+=6.0;
  }
 } else {
  s1+=12.0;
 }
} else {
 if(i21<525.07861328125){
  if(i18<597.75830078125){
   s0+=2.0;
  } else {
   s1+=1.0;
  }
 } else {
  if(i0<20785.693359375){
   s0+=5.0;
   s1+=31.0;
  } else {
   s1+=710.0;
  }
 }
}
if(i13<601.5966796875){
 if(i21<2052.33203125){
  if(i11<243.09619140625){
   s0+=1250.0;
   s1+=5.0;
  } else {
   s1+=5.0;
  }
 } else {
  s1+=18.0;
 }
} else {
 if(i21<501.94482421875){
  s0+=1.0;
 } else {
  if(i5<16161.38671875){
   s0+=1.0;
   s1+=15.0;
  } else {
   s1+=658.0;
  }
 }
}
if(i0<15328.71484375){
 if(i17<353.9287109375){
  s0+=1172.0;
 } else {
  if(i0<13594.130859375){
   s0+=13.0;
  } else {
   s1+=1.0;
  }
 }
} else {
 if(i3<98899.046875){
  if(i0<19761.16015625){
   s0+=5.0;
   s1+=25.0;
  } else {
   s1+=726.0;
  }
 } else {
  if(i19<21313.48046875){
   s1+=5.0;
  } else {
   s0+=6.0;
  }
 }
}
if(i13<294.052490234375){
 if(i20<21512.173828125){
  if(i13<251.856201171875){
   s0+=1154.0;
   s1+=3.0;
  } else {
   s0+=2.0;
   s1+=1.0;
  }
 } else {
  if(i4<21680.5703125){
   s0+=1.0;
  } else {
   s1+=10.0;
  }
 }
} else {
 if(i24<14606.0){
  if(i4<17642.841796875){
   s0+=6.0;
  } else {
   s0+=2.0;
   s1+=2.0;
  }
 } else {
  if(i6<14785.658203125){
   s0+=3.0;
   s1+=24.0;
  } else {
   s1+=745.0;
  }
 }
}
if(i13<601.5966796875){
 if(i7<25581.119140625){
  if(i0<20785.05859375){
   s0+=1209.0;
   s1+=4.0;
  } else {
   s1+=4.0;
  }
 } else {
  s1+=18.0;
 }
} else {
 if(i7<13610.546875){
  if(i19<7014.45947265625){
   s1+=7.0;
  } else {
   s0+=5.0;
  }
 } else {
  if(i4<20673.31640625){
   s0+=1.0;
   s1+=17.0;
  } else {
   s1+=688.0;
  }
 }
}
if(i17<591.87548828125){
 if(i11<243.09619140625){
  if(i0<18237.65625){
   s0+=1176.0;
   s1+=1.0;
  } else {
   s0+=3.0;
   s1+=9.0;
  }
 } else {
  s1+=15.0;
 }
} else {
 if(i15<69533.234375){
  if(i24<11540.0){
   s0+=3.0;
  } else {
   s0+=1.0;
   s1+=740.0;
  }
 } else {
  if(i16<33653.6640625){
   s0+=4.0;
  } else {
   s1+=1.0;
  }
 }
}
if(i10<351.6875){
 if(i20<21517.6875){
  if(i4<17969.22265625){
   s0+=1150.0;
   s1+=1.0;
  } else {
   s0+=16.0;
   s1+=12.0;
  }
 } else {
  s1+=24.0;
 }
} else {
 if(i7<13690.07421875){
  if(i19<7080.7822265625){
   s1+=14.0;
  } else {
   s0+=3.0;
  }
 } else {
  s1+=733.0;
 }
}
if(i0<16690.29296875){
 s0+=1172.0;
} else {
 if(i0<19950.96484375){
  if(i19<9425.69140625){
   s1+=18.0;
  } else {
   s0+=11.0;
  }
 } else {
  s1+=752.0;
 }
}
if(i24<14756.5){
 if(i24<12621.5){
  s0+=1162.0;
 } else {
  if(i19<7558.0791015625){
   s1+=2.0;
  } else {
   s0+=5.0;
  }
 }
} else {
 if(i0<21742.0390625){
  if(i5<22659.66015625){
   s0+=1.0;
   s1+=35.0;
  } else {
   s0+=2.0;
  }
 } else {
  s1+=746.0;
 }
}
if(i10<339.984375){
 if(i24<18317.5){
  s0+=1168.0;
 } else {
  s1+=26.0;
 }
} else {
 if(i21<493.3779296875){
  s0+=2.0;
 } else {
  if(i24<15422.5){
   s0+=1.0;
  } else {
   s0+=3.0;
   s1+=753.0;
  }
 }
}
if(i0<15362.06640625){
 if(i10<299.458984375){
  s0+=1189.0;
 } else {
  if(i6<12886.251953125){
   s1+=2.0;
  } else {
   s0+=5.0;
  }
 }
} else {
 if(i15<67307.96875){
  if(i24<11212.5){
   s0+=1.0;
  } else {
   s0+=3.0;
   s1+=746.0;
  }
 } else {
  if(i0<34736.69140625){
   s0+=4.0;
  } else {
   s1+=3.0;
  }
 }
}
if(i12<653.184814453125){
 if(i4<25166.28125){
  if(i24<17655.0){
   s0+=1191.0;
   s1+=1.0;
  } else {
   s1+=4.0;
  }
 } else {
  s1+=45.0;
 }
} else {
 if(i21<564.52001953125){
  if(i9<373.127685546875){
   s0+=7.0;
  } else {
   s0+=5.0;
   s1+=1.0;
  }
 } else {
  if(i5<16172.115234375){
   s0+=1.0;
   s1+=12.0;
  } else {
   s1+=686.0;
  }
 }
}
if(i10<339.48583984375){
 if(i2<4627.78564453125){
  if(i20<21563.09765625){
   s0+=1167.0;
   s1+=6.0;
  } else {
   s0+=2.0;
   s1+=8.0;
  }
 } else {
  s1+=13.0;
 }
} else {
 if(i21<525.07861328125){
  if(i16<11030.189453125){
   s1+=1.0;
  } else {
   s0+=2.0;
  }
 } else {
  s1+=754.0;
 }
}
if(i9<477.421875){
 if(i4<18052.328125){
  if(i24<12516.5){
   s0+=1181.0;
  } else {
   s0+=2.0;
   s1+=1.0;
  }
 } else {
  if(i16<26136.833984375){
   s1+=14.0;
  } else {
   s0+=20.0;
  }
 }
} else {
 if(i15<68871.71875){
  if(i13<-163.98681640625){
   s0+=3.0;
  } else {
   s0+=11.0;
   s1+=718.0;
  }
 } else {
  s0+=3.0;
 }
}
if(i13<437.02490234375){
 if(i24<17183.0){
  s0+=1229.0;
 } else {
  s1+=21.0;
 }
} else {
 if(i5<16892.064453125){
  if(i3<43804.4296875){
   s1+=27.0;
  } else {
   s0+=7.0;
  }
 } else {
  if(i23<8110.4443359375){
   s0+=1.0;
   s1+=26.0;
  } else {
   s1+=642.0;
  }
 }
}
if(i1<143.15576171875){
 if(i24<14606.0){
  if(i10<286.77099609375){
   s0+=1150.0;
  } else {
   s0+=4.0;
   s1+=3.0;
  }
 } else {
  if(i19<21446.404296875){
   s0+=1.0;
   s1+=241.0;
  } else {
   s0+=2.0;
   s1+=8.0;
  }
 }
} else {
 if(i10<200.053466796875){
  if(i0<16536.171875){
   s0+=33.0;
  } else {
   s1+=7.0;
  }
 } else {
  if(i24<20595.5){
   s0+=1.0;
   s1+=8.0;
  } else {
   s1+=495.0;
  }
 }
}
if(i24<14756.5){
 if(i10<650.46337890625){
  s0+=1190.0;
 } else {
  s1+=1.0;
 }
} else {
 if(i24<20013.0){
  if(i24<19917.0){
   s0+=1.0;
   s1+=37.0;
  } else {
   s0+=1.0;
  }
 } else {
  s1+=723.0;
 }
}
if(i17<163.171875){
 if(i20<21952.56640625){
  if(i0<17871.91796875){
   s0+=1203.0;
  } else {
   s0+=1.0;
   s1+=5.0;
  }
 } else {
  s1+=5.0;
 }
} else {
 if(i0<19005.24609375){
  if(i14<17835.92578125){
   s1+=16.0;
  } else {
   s0+=33.0;
  }
 } else {
  if(i3<101045.359375){
   s0+=1.0;
   s1+=678.0;
  } else {
   s0+=2.0;
   s1+=9.0;
  }
 }
}
if(i0<17111.6796875){
 if(i12<1096.149658203125){
  if(i24<11716.0){
   s0+=1201.0;
  } else {
   s0+=6.0;
   s1+=3.0;
  }
 } else {
  s1+=3.0;
 }
} else {
 if(i15<66680.34375){
  if(i9<274.9853515625){
   s0+=4.0;
   s1+=11.0;
  } else {
   s0+=1.0;
   s1+=719.0;
  }
 } else {
  if(i21<1953.5263671875){
   s0+=3.0;
  } else {
   s1+=2.0;
  }
 }
}
if(i24<16080.0){
 if(i13<917.94091796875){
  if(i13<509.3037109375){
   s0+=1201.0;
  } else {
   s0+=5.0;
   s1+=1.0;
  }
 } else {
  if(i21<518.56396484375){
   s1+=2.0;
  } else {
   s0+=2.0;
  }
 }
} else {
 if(i20<10380.75390625){
  s0+=1.0;
 } else {
  if(i10<339.984375){
   s0+=2.0;
   s1+=24.0;
  } else {
   s0+=1.0;
   s1+=714.0;
  }
 }
}
if(i13<626.900390625){
 if(i24<15608.0){
  if(i18<215.76953125){
   s0+=1136.0;
  } else {
   s0+=61.0;
   s1+=1.0;
  }
 } else {
  s1+=32.0;
 }
} else {
 if(i24<16914.5){
  if(i23<10177.439453125){
   s0+=10.0;
   s1+=3.0;
  } else {
   s1+=5.0;
  }
 } else {
  s1+=705.0;
 }
}
if(i4<18245.125){
 if(i24<11716.0){
  s0+=1145.0;
 } else {
  if(i9<185.7802734375){
   s0+=5.0;
  } else {
   s0+=2.0;
   s1+=22.0;
  }
 }
} else {
 if(i24<16466.5){
  s0+=16.0;
 } else {
  if(i3<100123.1640625){
   s1+=751.0;
  } else {
   s0+=1.0;
   s1+=11.0;
  }
 }
}
if(i17<271.58740234375){
 if(i2<4825.7861328125){
  if(i5<25868.67578125){
   s0+=1144.0;
   s1+=3.0;
  } else {
   s1+=2.0;
  }
 } else {
  if(i24<15896.0){
   s0+=1.0;
  } else {
   s1+=6.0;
  }
 }
} else {
 if(i0<13788.8642578125){
  s0+=22.0;
 } else {
  if(i15<71321.4453125){
   s0+=6.0;
   s1+=765.0;
  } else {
   s0+=4.0;
  }
 }
}
if(i10<337.60693359375){
 if(i0<19809.404296875){
  if(i0<17871.91796875){
   s0+=1172.0;
  } else {
   s0+=4.0;
   s1+=2.0;
  }
 } else {
  s1+=30.0;
 }
} else {
 if(i24<16045.0){
  s0+=5.0;
 } else {
  if(i24<17977.5){
   s0+=2.0;
   s1+=10.0;
  } else {
   s1+=728.0;
  }
 }
}
if(i13<505.9228515625){
 if(i7<25581.119140625){
  if(i24<15608.0){
   s0+=1193.0;
  } else {
   s1+=8.0;
  }
 } else {
  s1+=15.0;
 }
} else {
 if(i20<10151.7138671875){
  s0+=1.0;
 } else {
  if(i7<13690.07421875){
   s0+=3.0;
   s1+=19.0;
  } else {
   s0+=2.0;
   s1+=712.0;
  }
 }
}
if(i10<339.984375){
 if(i21<1896.3076171875){
  if(i7<19523.125){
   s0+=1117.0;
   s1+=1.0;
  } else {
   s0+=64.0;
   s1+=7.0;
  }
 } else {
  s1+=15.0;
 }
} else {
 if(i0<15505.830078125){
  s0+=1.0;
 } else {
  if(i12<3225.2685546875){
   s0+=1.0;
   s1+=202.0;
  } else {
   s1+=545.0;
  }
 }
}
if(i11<241.25244140625){
 if(i24<17857.0){
  if(i10<529.8515625){
   s0+=1205.0;
   s1+=2.0;
  } else {
   s0+=1.0;
   s1+=2.0;
  }
 } else {
  s1+=75.0;
 }
} else {
 if(i24<14606.0){
  s0+=1.0;
 } else {
  if(i21<592.2431640625){
   s0+=1.0;
   s1+=9.0;
  } else {
   s1+=657.0;
  }
 }
}
if(i4<18023.34375){
 if(i8<298.72265625){
  s0+=1104.0;
 } else {
  if(i19<7125.8984375){
   s1+=9.0;
  } else {
   s0+=55.0;
  }
 }
} else {
 if(i9<249.5849609375){
  if(i3<66578.8125){
   s1+=17.0;
  } else {
   s0+=18.0;
  }
 } else {
  if(i5<16172.115234375){
   s0+=1.0;
  } else {
   s0+=4.0;
   s1+=745.0;
  }
 }
}
if(i12<718.29541015625){
 if(i24<16080.0){
  if(i13<509.3037109375){
   s0+=1211.0;
  } else {
   s0+=5.0;
   s1+=2.0;
  }
 } else {
  s1+=72.0;
 }
} else {
 if(i24<11715.0){
  s0+=4.0;
 } else {
  if(i7<13594.548828125){
   s0+=1.0;
   s1+=18.0;
  } else {
   s1+=640.0;
  }
 }
}
if(i24<14756.5){
 if(i4<17951.0859375){
  s0+=1171.0;
 } else {
  if(i17<855.79443359375){
   s0+=14.0;
  } else {
   s1+=1.0;
  }
 }
} else {
 if(i20<10387.65234375){
  s0+=2.0;
 } else {
  if(i0<18722.72265625){
   s0+=2.0;
   s1+=15.0;
  } else {
   s1+=748.0;
  }
 }
}
if(i24<15573.0){
 if(i9<873.61572265625){
  if(i11<158.954833984375){
   s0+=1217.0;
  } else {
   s0+=14.0;
   s1+=1.0;
  }
 } else {
  s1+=1.0;
 }
} else {
 if(i24<20048.0){
  if(i15<30745.177734375){
   s1+=30.0;
  } else {
   s0+=5.0;
   s1+=2.0;
  }
 } else {
  s1+=683.0;
 }
}
if(i0<17136.919921875){
 if(i13<509.3037109375){
  s0+=1149.0;
 } else {
  if(i20<10796.0302734375){
   s0+=3.0;
  } else {
   s1+=9.0;
  }
 }
} else {
 if(i18<620.2587890625){
  if(i15<33194.69921875){
   s0+=1.0;
   s1+=53.0;
  } else {
   s0+=8.0;
   s1+=3.0;
  }
 } else {
  if(i1<-1962.32421875){
   s0+=1.0;
   s1+=1.0;
  } else {
   s0+=1.0;
   s1+=724.0;
  }
 }
}
if(i9<561.46826171875){
 if(i0<18402.638671875){
  if(i11<300.30908203125){
   s0+=1212.0;
  } else {
   s1+=1.0;
  }
 } else {
  if(i3<77398.0859375){
   s1+=16.0;
  } else {
   s0+=1.0;
  }
 }
} else {
 if(i9<1033.67919921875){
  if(i14<17743.396484375){
   s1+=23.0;
  } else {
   s0+=10.0;
   s1+=7.0;
  }
 } else {
  if(i24<16105.5){
   s0+=1.0;
   s1+=2.0;
  } else {
   s1+=680.0;
  }
 }
}
if(i4<18052.328125){
 if(i10<339.48583984375){
  s0+=1164.0;
 } else {
  if(i3<69817.5078125){
   s1+=19.0;
  } else {
   s0+=1.0;
  }
 }
} else {
 if(i17<513.798828125){
  if(i2<-5475.28173828125){
   s0+=18.0;
  } else {
   s1+=17.0;
  }
 } else {
  if(i23<8110.4443359375){
   s0+=3.0;
   s1+=24.0;
  } else {
   s1+=707.0;
  }
 }
}
if(i13<295.549072265625){
 if(i21<1997.2705078125){
  if(i10<170.717529296875){
   s0+=1201.0;
   s1+=3.0;
  } else {
   s1+=1.0;
  }
 } else {
  s1+=12.0;
 }
} else {
 if(i10<339.19677734375){
  if(i20<18683.2421875){
   s0+=6.0;
   s1+=1.0;
  } else {
   s1+=7.0;
  }
 } else {
  if(i7<13594.548828125){
   s0+=5.0;
   s1+=15.0;
  } else {
   s0+=1.0;
   s1+=701.0;
  }
 }
}
if(i13<311.721923828125){
 if(i23<13204.21484375){
  if(i0<19774.89453125){
   s0+=1197.0;
   s1+=1.0;
  } else {
   s1+=5.0;
  }
 } else {
  s1+=9.0;
 }
} else {
 if(i21<500.80126953125){
  s0+=8.0;
 } else {
  if(i21<577.966796875){
   s0+=3.0;
   s1+=14.0;
  } else {
   s0+=1.0;
   s1+=715.0;
  }
 }
}
if(i4<18286.89453125){
 if(i11<299.08251953125){
  if(i13<509.3037109375){
   s0+=1165.0;
  } else {
   s0+=2.0;
   s1+=3.0;
  }
 } else {
  if(i18<513.64990234375){
   s1+=17.0;
  } else {
   s0+=6.0;
  }
 }
} else {
 if(i15<62620.66796875){
  if(i15<53600.02734375){
   s1+=692.0;
  } else {
   s0+=10.0;
   s1+=39.0;
  }
 } else {
  if(i4<29730.63671875){
   s0+=13.0;
  } else {
   s1+=6.0;
  }
 }
}
if(i10<350.2734375){
 if(i8<970.1494140625){
  if(i0<20418.77734375){
   s0+=1206.0;
   s1+=2.0;
  } else {
   s0+=2.0;
   s1+=16.0;
  }
 } else {
  if(i3<74060.9296875){
   s1+=10.0;
  } else {
   s0+=1.0;
  }
 }
} else {
 if(i24<17082.0){
  if(i18<741.2060546875){
   s0+=2.0;
  } else {
   s1+=1.0;
  }
 } else {
  if(i20<10380.75390625){
   s0+=1.0;
  } else {
   s1+=712.0;
  }
 }
}
if(i17<561.568359375){
 if(i17<149.32763671875){
  if(i24<15869.0){
   s0+=1145.0;
  } else {
   s1+=11.0;
  }
 } else {
  if(i2<-6275.96484375){
   s0+=30.0;
  } else {
   s1+=20.0;
  }
 }
} else {
 if(i13<605.51953125){
  if(i4<27081.892578125){
   s0+=3.0;
  } else {
   s1+=15.0;
  }
 } else {
  if(i21<529.53466796875){
   s0+=1.0;
   s1+=2.0;
  } else {
   s1+=726.0;
  }
 }
}
if(i18<321.0732421875){
 if(i24<15635.0){
  if(i13<436.45166015625){
   s0+=1170.0;
  } else {
   s1+=1.0;
  }
 } else {
  s1+=25.0;
 }
} else {
 if(i24<16105.5){
  if(i19<9323.21484375){
   s1+=1.0;
  } else {
   s0+=28.0;
  }
 } else {
  if(i15<62620.66796875){
   s1+=721.0;
  } else {
   s0+=1.0;
   s1+=6.0;
  }
 }
}
if(i9<507.30859375){
 if(i11<243.09619140625){
  if(i1<352.02734375){
   s0+=1171.0;
   s1+=6.0;
  } else {
   s0+=3.0;
   s1+=3.0;
  }
 } else {
  s1+=14.0;
 }
} else {
 if(i0<15821.921875){
  s0+=8.0;
 } else {
  if(i20<10380.75390625){
   s0+=1.0;
  } else {
   s0+=1.0;
   s1+=746.0;
  }
 }
}
if(i21<1097.25732421875){
 if(i4<16210.2685546875){
  if(i24<15573.0){
   s0+=1109.0;
   s1+=1.0;
  } else {
   s1+=6.0;
  }
 } else {
  if(i19<8966.0625){
   s1+=105.0;
  } else {
   s0+=8.0;
   s1+=1.0;
  }
 }
} else {
 if(i18<253.81640625){
  if(i14<28573.45703125){
   s1+=5.0;
  } else {
   s0+=48.0;
  }
 } else {
  if(i4<20696.53125){
   s0+=5.0;
   s1+=2.0;
  } else {
   s0+=3.0;
   s1+=660.0;
  }
 }
}
if(i10<152.692626953125){
 if(i23<13200.388671875){
  if(i6<19287.1796875){
   s0+=1133.0;
  } else {
   s0+=47.0;
   s1+=3.0;
  }
 } else {
  s1+=9.0;
 }
} else {
 if(i13<626.900390625){
  if(i1<-236.126953125){
   s0+=10.0;
  } else {
   s0+=1.0;
   s1+=13.0;
  }
 } else {
  if(i21<529.53466796875){
   s0+=2.0;
   s1+=1.0;
  } else {
   s0+=2.0;
   s1+=732.0;
  }
 }
}
if(i13<605.51953125){
 if(i0<21947.22265625){
  if(i13<509.3037109375){
   s0+=1155.0;
  } else {
   s0+=4.0;
   s1+=1.0;
  }
 } else {
  s1+=18.0;
 }
} else {
 if(i20<10380.75390625){
  s0+=2.0;
 } else {
  if(i0<19761.16015625){
   s0+=7.0;
   s1+=19.0;
  } else {
   s1+=747.0;
  }
 }
}
if(i13<595.49609375){
 if(i6<26064.150390625){
  if(i1<473.48388671875){
   s0+=1167.0;
   s1+=5.0;
  } else {
   s1+=3.0;
  }
 } else {
  s1+=16.0;
 }
} else {
 if(i13<1558.99853515625){
  if(i21<525.07861328125){
   s0+=1.0;
  } else {
   s0+=3.0;
   s1+=63.0;
  }
 } else {
  s1+=695.0;
 }
}
if(i10<339.984375){
 if(i24<15441.5){
  s0+=1236.0;
 } else {
  if(i10<339.1015625){
   s1+=25.0;
  } else {
   s0+=2.0;
  }
 }
} else {
 if(i24<16045.0){
  s0+=2.0;
 } else {
  if(i7<13580.876953125){
   s0+=1.0;
   s1+=18.0;
  } else {
   s1+=669.0;
  }
 }
}
if(i24<14606.0){
 if(i9<930.6484375){
  if(i8<356.9443359375){
   s0+=1117.0;
  } else {
   s0+=48.0;
   s1+=1.0;
  }
 } else {
  s1+=2.0;
 }
} else {
 if(i14<40490.890625){
  s1+=768.0;
 } else {
  if(i21<1748.529296875){
   s0+=1.0;
  } else {
   s1+=16.0;
  }
 }
}
if(i8<718.6142578125){
 if(i10<204.46435546875){
  if(i24<16890.0){
   s0+=1149.0;
  } else {
   s1+=8.0;
  }
 } else {
  if(i0<13788.8642578125){
   s0+=3.0;
  } else {
   s1+=29.0;
  }
 }
} else {
 if(i10<344.2216796875){
  if(i21<1921.87109375){
   s0+=19.0;
  } else {
   s1+=18.0;
  }
 } else {
  if(i5<16162.72265625){
   s0+=5.0;
   s1+=2.0;
  } else {
   s0+=1.0;
   s1+=719.0;
  }
 }
}
if(i24<16045.0){
 if(i11<158.391357421875){
  s0+=1155.0;
 } else {
  if(i0<11881.0390625){
   s0+=16.0;
  } else {
   s1+=2.0;
  }
 }
} else {
 if(i20<10555.91796875){
  if(i0<22211.609375){
   s0+=2.0;
  } else {
   s1+=8.0;
  }
 } else {
  if(i10<340.2734375){
   s0+=1.0;
   s1+=28.0;
  } else {
   s1+=741.0;
  }
 }
}
if(i12<718.29541015625){
 if(i5<24284.51171875){
  if(i13<1038.14990234375){
   s0+=1190.0;
   s1+=5.0;
  } else {
   s0+=2.0;
   s1+=3.0;
  }
 } else {
  if(i0<20688.6796875){
   s0+=2.0;
  } else {
   s1+=65.0;
  }
 }
} else {
 if(i23<7132.0361328125){
  s0+=3.0;
 } else {
  if(i21<592.2431640625){
   s0+=4.0;
   s1+=7.0;
  } else {
   s1+=672.0;
  }
 }
}
if(i17<591.9404296875){
 if(i0<18036.8984375){
  if(i0<15328.71484375){
   s0+=1167.0;
  } else {
   s0+=4.0;
   s1+=2.0;
  }
 } else {
  if(i2<-6884.68994140625){
   s0+=2.0;
  } else {
   s1+=17.0;
  }
 }
} else {
 if(i24<16069.5){
  if(i12<2218.62255859375){
   s0+=11.0;
  } else {
   s1+=1.0;
  }
 } else {
  s1+=749.0;
 }
}
if(i24<14756.5){
 if(i24<12647.5){
  s0+=1175.0;
 } else {
  if(i7<12992.4140625){
   s1+=2.0;
  } else {
   s0+=8.0;
  }
 }
} else {
 if(i0<20740.666015625){
  if(i3<56418.390625){
   s1+=27.0;
  } else {
   s0+=3.0;
   s1+=3.0;
  }
 } else {
  s1+=735.0;
 }
}
if(i24<14606.0){
 if(i10<650.46337890625){
  s0+=1171.0;
 } else {
  if(i8<1004.53271484375){
   s0+=2.0;
  } else {
   s1+=1.0;
  }
 }
} else {
 if(i12<-279.3115234375){
  if(i24<18595.0){
   s0+=5.0;
  } else {
   s1+=41.0;
  }
 } else {
  if(i20<10537.013671875){
   s0+=1.0;
   s1+=7.0;
  } else {
   s1+=725.0;
  }
 }
}
if(i9<357.52587890625){
 if(i24<15608.0){
  s0+=1175.0;
 } else {
  s1+=19.0;
 }
} else {
 if(i0<13788.8642578125){
  s0+=18.0;
 } else {
  if(i24<20152.0){
   s0+=6.0;
   s1+=28.0;
  } else {
   s1+=707.0;
  }
 }
}
if(i13<505.9228515625){
 if(i21<2052.33203125){
  if(i12<793.73193359375){
   s0+=1223.0;
   s1+=4.0;
  } else {
   s1+=4.0;
  }
 } else {
  s1+=17.0;
 }
} else {
 if(i21<493.3779296875){
  s0+=2.0;
 } else {
  if(i20<10537.013671875){
   s0+=2.0;
   s1+=2.0;
  } else {
   s0+=5.0;
   s1+=694.0;
  }
 }
}
if(i24<14756.5){
 if(i10<650.46337890625){
  s0+=1196.0;
 } else {
  s1+=1.0;
 }
} else {
 if(i5<16162.72265625){
  if(i7<13492.142578125){
   s1+=15.0;
  } else {
   s0+=3.0;
  }
 } else {
  if(i24<16080.0){
   s0+=1.0;
   s1+=1.0;
  } else {
   s0+=1.0;
   s1+=735.0;
  }
 }
}
if(i24<15457.5){
 s0+=1198.0;
} else {
 s1+=755.0;
}
if(i24<14756.5){
 s0+=1217.0;
} else {
 if(i5<16221.46875){
  if(i9<925.03955078125){
   s1+=12.0;
  } else {
   s0+=4.0;
  }
 } else {
  if(i0<21801.052734375){
   s0+=2.0;
   s1+=15.0;
  } else {
   s1+=703.0;
  }
 }
}
if(i24<14756.5){
 if(i10<299.458984375){
  s0+=1179.0;
 } else {
  if(i5<14389.390625){
   s1+=2.0;
  } else {
   s0+=4.0;
   s1+=1.0;
  }
 }
} else {
 if(i21<592.2431640625){
  if(i20<10662.5625){
   s0+=2.0;
  } else {
   s1+=14.0;
  }
 } else {
  if(i9<572.783203125){
   s0+=1.0;
   s1+=21.0;
  } else {
   s1+=729.0;
  }
 }
}
if(i4<18247.16796875){
 if(i0<13911.095703125){
  s0+=1177.0;
 } else {
  if(i19<8976.2109375){
   s1+=14.0;
  } else {
   s0+=8.0;
   s1+=2.0;
  }
 }
} else {
 if(i24<14522.5){
  s0+=15.0;
 } else {
  if(i3<101045.359375){
   s1+=722.0;
  } else {
   s0+=1.0;
   s1+=14.0;
  }
 }
}
if(i11<202.0361328125){
 if(i24<15633.5){
  if(i9<821.97412109375){
   s0+=1198.0;
   s1+=2.0;
  } else {
   s1+=1.0;
  }
 } else {
  s1+=72.0;
 }
} else {
 if(i0<12895.4267578125){
  s0+=5.0;
 } else {
  if(i21<525.07861328125){
   s0+=2.0;
  } else {
   s0+=2.0;
   s1+=671.0;
  }
 }
}
if(i9<692.2041015625){
 if(i12<793.73193359375){
  if(i0<17696.771484375){
   s0+=1185.0;
  } else {
   s0+=2.0;
   s1+=9.0;
  }
 } else {
  if(i24<12344.5){
   s0+=1.0;
  } else {
   s1+=22.0;
  }
 }
} else {
 if(i21<594.23779296875){
  if(i18<622.63623046875){
   s0+=5.0;
  } else {
   s1+=3.0;
  }
 } else {
  if(i17<816.25341796875){
   s0+=3.0;
   s1+=22.0;
  } else {
   s1+=701.0;
  }
 }
}
if(i0<17546.921875){
 if(i10<364.2861328125){
  s0+=1200.0;
 } else {
  s1+=5.0;
 }
} else {
 if(i9<436.2578125){
  if(i3<78784.6875){
   s1+=21.0;
  } else {
   s0+=6.0;
  }
 } else {
  if(i5<15891.41015625){
   s0+=1.0;
   s1+=9.0;
  } else {
   s1+=711.0;
  }
 }
}
if(i18<574.16015625){
 if(i17<420.03173828125){
  if(i12<793.73193359375){
   s0+=1186.0;
   s1+=5.0;
  } else {
   s1+=7.0;
  }
 } else {
  if(i1<-244.625){
   s0+=16.0;
   s1+=1.0;
  } else {
   s1+=34.0;
  }
 }
} else {
 if(i17<939.7890625){
  if(i0<22702.0625){
   s0+=9.0;
   s1+=3.0;
  } else {
   s1+=7.0;
  }
 } else {
  s1+=685.0;
 }
}
if(i10<339.19677734375){
 if(i7<25987.14453125){
  if(i0<18237.65625){
   s0+=1190.0;
  } else {
   s0+=5.0;
   s1+=6.0;
  }
 } else {
  s1+=18.0;
 }
} else {
 if(i21<493.3779296875){
  s0+=1.0;
 } else {
  if(i23<8110.4443359375){
   s0+=1.0;
   s1+=26.0;
  } else {
   s1+=706.0;
  }
 }
}
if(i4<17403.64453125){
 if(i13<765.9306640625){
  if(i10<299.458984375){
   s0+=1143.0;
  } else {
   s0+=6.0;
   s1+=2.0;
  }
 } else {
  if(i21<538.31201171875){
   s0+=1.0;
  } else {
   s1+=9.0;
  }
 }
} else {
 if(i24<14522.5){
  s0+=19.0;
 } else {
  if(i24<18317.5){
   s0+=3.0;
   s1+=14.0;
  } else {
   s1+=756.0;
  }
 }
}
if(i5<17578.416015625){
 if(i4<16185.271484375){
  if(i10<299.458984375){
   s0+=1187.0;
  } else {
   s0+=2.0;
   s1+=7.0;
  }
 } else {
  if(i23<9123.607421875){
   s1+=27.0;
  } else {
   s0+=5.0;
   s1+=1.0;
  }
 }
} else {
 if(i0<20351.03515625){
  if(i2<-7562.962890625){
   s0+=64.0;
  } else {
   s1+=4.0;
  }
 } else {
  if(i17<528.421875){
   s0+=2.0;
   s1+=22.0;
  } else {
   s1+=632.0;
  }
 }
}
if(i9<525.1435546875){
 if(i11<228.49462890625){
  if(i6<27083.626953125){
   s0+=1201.0;
   s1+=6.0;
  } else {
   s1+=2.0;
  }
 } else {
  s1+=13.0;
 }
} else {
 if(i4<18340.7734375){
  if(i14<16177.7939453125){
   s1+=15.0;
  } else {
   s0+=6.0;
  }
 } else {
  if(i15<69533.234375){
   s0+=2.0;
   s1+=706.0;
  } else {
   s0+=1.0;
   s1+=1.0;
  }
 }
}
if(i24<16069.5){
 if(i13<1451.67724609375){
  if(i24<12407.5){
   s0+=1182.0;
  } else {
   s0+=12.0;
   s1+=2.0;
  }
 } else {
  s1+=1.0;
 }
} else {
 if(i24<17857.0){
  if(i3<60600.22265625){
   s1+=5.0;
  } else {
   s0+=5.0;
  }
 } else {
  if(i5<16172.115234375){
   s0+=3.0;
   s1+=10.0;
  } else {
   s1+=733.0;
  }
 }
}
if(i13<605.51953125){
 if(i6<25544.33203125){
  if(i2<4825.7861328125){
   s0+=1200.0;
   s1+=3.0;
  } else {
   s0+=1.0;
   s1+=6.0;
  }
 } else {
  s1+=18.0;
 }
} else {
 s1+=725.0;
}
if(i4<17951.0859375){
 if(i10<339.48583984375){
  s0+=1148.0;
 } else {
  s1+=20.0;
 }
} else {
 if(i10<339.984375){
  if(i3<60545.0859375){
   s1+=33.0;
  } else {
   s0+=26.0;
  }
 } else {
  if(i4<20618.642578125){
   s0+=1.0;
   s1+=24.0;
  } else {
   s1+=701.0;
  }
 }
}
if(i4<18067.646484375){
 if(i13<606.61962890625){
  if(i1<535.21435546875){
   s0+=1210.0;
   s1+=1.0;
  } else {
   s1+=1.0;
  }
 } else {
  if(i18<489.15087890625){
   s1+=12.0;
  } else {
   s0+=3.0;
  }
 }
} else {
 if(i19<21306.96484375){
  if(i5<16162.72265625){
   s0+=1.0;
  } else {
   s0+=1.0;
   s1+=684.0;
  }
 } else {
  if(i22<17313.3671875){
   s0+=16.0;
  } else {
   s1+=24.0;
  }
 }
}
if(i18<321.54345703125){
 if(i1<370.86669921875){
  if(i24<15406.5){
   s0+=1183.0;
   s1+=1.0;
  } else {
   s1+=10.0;
  }
 } else {
  if(i10<-372.02490234375){
   s0+=1.0;
  } else {
   s1+=16.0;
  }
 }
} else {
 if(i17<595.46337890625){
  if(i20<18711.11328125){
   s0+=28.0;
   s1+=1.0;
  } else {
   s1+=8.0;
  }
 } else {
  if(i4<16101.578125){
   s0+=4.0;
  } else {
   s0+=5.0;
   s1+=696.0;
  }
 }
}
if(i13<589.33984375){
 if(i4<19414.8125){
  if(i10<173.07958984375){
   s0+=1150.0;
  } else {
   s0+=10.0;
   s1+=2.0;
  }
 } else {
  if(i16<27202.6640625){
   s1+=28.0;
  } else {
   s0+=8.0;
  }
 }
} else {
 if(i21<501.94482421875){
  s0+=1.0;
 } else {
  if(i0<20785.693359375){
   s0+=2.0;
   s1+=26.0;
  } else {
   s1+=726.0;
  }
 }
}
if(i9<507.30859375){
 if(i5<22066.267578125){
  if(i11<243.09619140625){
   s0+=1186.0;
   s1+=7.0;
  } else {
   s1+=4.0;
  }
 } else {
  if(i0<20026.8359375){
   s0+=4.0;
  } else {
   s1+=13.0;
  }
 }
} else {
 if(i21<495.43017578125){
  s0+=4.0;
 } else {
  if(i17<897.62548828125){
   s0+=9.0;
   s1+=28.0;
  } else {
   s1+=698.0;
  }
 }
}
if(i0<17386.15625){
 if(i11<300.30908203125){
  if(i12<588.8447265625){
   s0+=1170.0;
  } else {
   s0+=15.0;
   s1+=2.0;
  }
 } else {
  s1+=2.0;
 }
} else {
 if(i0<20785.693359375){
  if(i19<9182.216796875){
   s1+=20.0;
  } else {
   s0+=12.0;
  }
 } else {
  if(i3<101045.359375){
   s1+=726.0;
  } else {
   s0+=1.0;
   s1+=5.0;
  }
 }
}
if(i24<16045.0){
 if(i24<11719.0){
  s0+=1170.0;
 } else {
  if(i19<9948.193359375){
   s0+=2.0;
   s1+=3.0;
  } else {
   s0+=14.0;
  }
 }
} else {
 if(i20<10380.75390625){
  s0+=1.0;
 } else {
  if(i20<10547.4384765625){
   s0+=1.0;
   s1+=3.0;
  } else {
   s0+=1.0;
   s1+=758.0;
  }
 }
}
if(i24<14756.5){
 if(i17<855.79443359375){
  if(i18<214.841796875){
   s0+=1123.0;
  } else {
   s0+=63.0;
   s1+=1.0;
  }
 } else {
  s1+=2.0;
 }
} else {
 if(i9<998.974609375){
  if(i3<58556.66796875){
   s1+=54.0;
  } else {
   s0+=4.0;
   s1+=1.0;
  }
 } else {
  s1+=705.0;
 }
}
if(i24<14606.0){
 if(i24<12647.5){
  s0+=1199.0;
 } else {
  if(i6<13722.765625){
   s1+=1.0;
  } else {
   s0+=10.0;
  }
 }
} else {
 if(i23<8111.68896484375){
  if(i15<38840.7265625){
   s1+=26.0;
  } else {
   s0+=1.0;
  }
 } else {
  s1+=716.0;
 }
}
if(i17<271.58740234375){
 if(i6<25579.38671875){
  if(i0<20418.77734375){
   s0+=1167.0;
  } else {
   s1+=8.0;
  }
 } else {
  s1+=4.0;
 }
} else {
 if(i0<13788.8642578125){
  s0+=18.0;
 } else {
  if(i24<16914.5){
   s0+=9.0;
   s1+=4.0;
  } else {
   s0+=1.0;
   s1+=742.0;
  }
 }
}
if(i13<288.748046875){
 if(i24<15608.0){
  s0+=1217.0;
 } else {
  s1+=20.0;
 }
} else {
 if(i24<16045.0){
  s0+=15.0;
 } else {
  if(i20<10380.75390625){
   s0+=1.0;
  } else {
   s0+=2.0;
   s1+=698.0;
  }
 }
}
if(i8<696.8330078125){
 if(i11<299.08251953125){
  if(i0<20850.24609375){
   s0+=1223.0;
   s1+=2.0;
  } else {
   s1+=8.0;
  }
 } else {
  s1+=13.0;
 }
} else {
 if(i0<17298.78125){
  s0+=12.0;
 } else {
  if(i4<18247.16796875){
   s0+=5.0;
   s1+=7.0;
  } else {
   s0+=1.0;
   s1+=682.0;
  }
 }
}
if(i0<17386.15625){
 if(i2<5743.5322265625){
  if(i4<17896.96484375){
   s0+=1151.0;
   s1+=1.0;
  } else {
   s0+=21.0;
   s1+=1.0;
  }
 } else {
  s1+=2.0;
 }
} else {
 if(i24<16956.5){
  if(i21<548.21728515625){
   s0+=1.0;
   s1+=2.0;
  } else {
   s0+=9.0;
  }
 } else {
  if(i3<101045.359375){
   s1+=757.0;
  } else {
   s0+=1.0;
   s1+=7.0;
  }
 }
}
if(i10<339.19677734375){
 if(i24<15608.0){
  s0+=1218.0;
 } else {
  s1+=32.0;
 }
} else {
 if(i24<16914.5){
  if(i19<9425.69140625){
   s1+=8.0;
  } else {
   s0+=7.0;
   s1+=2.0;
  }
 } else {
  if(i5<16162.72265625){
   s0+=1.0;
   s1+=15.0;
  } else {
   s1+=670.0;
  }
 }
}
if(i17<591.87548828125){
 if(i2<4825.7861328125){
  if(i23<13369.203125){
   s0+=1215.0;
   s1+=6.0;
  } else {
   s1+=9.0;
  }
 } else {
  if(i3<20536.419921875){
   s0+=1.0;
  } else {
   s1+=21.0;
  }
 }
} else {
 if(i4<16185.271484375){
  s0+=2.0;
 } else {
  if(i20<10385.931640625){
   s0+=1.0;
  } else {
   s0+=3.0;
   s1+=695.0;
  }
 }
}
if(i0<13788.8642578125){
 s0+=1152.0;
} else {
 if(i19<26157.359375){
  if(i19<21306.96484375){
   s0+=5.0;
   s1+=749.0;
  } else {
   s0+=9.0;
   s1+=32.0;
  }
 } else {
  s0+=6.0;
 }
}
if(i17<468.14111328125){
 if(i23<13369.203125){
  if(i2<4660.69384765625){
   s0+=1177.0;
   s1+=5.0;
  } else {
   s0+=3.0;
   s1+=10.0;
  }
 } else {
  s1+=6.0;
 }
} else {
 if(i4<16185.04296875){
  if(i11<95.6279296875){
   s0+=7.0;
  } else {
   s1+=2.0;
  }
 } else {
  if(i17<923.2783203125){
   s0+=5.0;
   s1+=41.0;
  } else {
   s0+=1.0;
   s1+=696.0;
  }
 }
}
if(i0<16224.287109375){
 if(i10<299.458984375){
  s0+=1191.0;
 } else {
  if(i23<9023.0283203125){
   s1+=2.0;
  } else {
   s0+=4.0;
  }
 }
} else {
 if(i15<62620.66796875){
  if(i24<16914.5){
   s0+=7.0;
   s1+=8.0;
  } else {
   s1+=729.0;
  }
 } else {
  if(i22<17169.322265625){
   s0+=8.0;
  } else {
   s1+=4.0;
  }
 }
}
if(i24<14756.5){
 if(i8<1041.67822265625){
  s0+=1199.0;
 } else {
  if(i21<813.9775390625){
   s1+=3.0;
  } else {
   s0+=3.0;
  }
 }
} else {
 if(i0<21742.0390625){
  if(i20<10486.67578125){
   s0+=3.0;
  } else {
   s0+=1.0;
   s1+=40.0;
  }
 } else {
  s1+=704.0;
 }
}
if(i24<14606.0){
 if(i9<930.6484375){
  s0+=1184.0;
 } else {
  s1+=2.0;
 }
} else {
 if(i6<14792.0615234375){
  if(i16<10187.330078125){
   s1+=17.0;
  } else {
   s0+=3.0;
  }
 } else {
  if(i11<-152.5322265625){
   s0+=1.0;
   s1+=41.0;
  } else {
   s1+=705.0;
  }
 }
}
if(i24<14756.5){
 if(i18<622.42236328125){
  if(i10<286.77099609375){
   s0+=1193.0;
  } else {
   s0+=7.0;
   s1+=1.0;
  }
 } else {
  if(i14<28452.080078125){
   s1+=2.0;
  } else {
   s0+=4.0;
  }
 }
} else {
 if(i4<18067.646484375){
  if(i17<858.08447265625){
   s1+=15.0;
  } else {
   s0+=1.0;
  }
 } else {
  s1+=730.0;
 }
}
if(i24<15573.0){
 if(i13<509.3037109375){
  s0+=1188.0;
 } else {
  if(i1<-192.890625){
   s0+=7.0;
   s1+=1.0;
  } else {
   s1+=4.0;
  }
 }
} else {
 if(i20<10380.75390625){
  s0+=1.0;
 } else {
  if(i24<17857.0){
   s0+=3.0;
   s1+=10.0;
  } else {
   s1+=739.0;
  }
 }
}
if(i24<16045.0){
 if(i13<509.3037109375){
  s0+=1203.0;
 } else {
  if(i19<10023.912109375){
   s0+=1.0;
   s1+=3.0;
  } else {
   s0+=5.0;
  }
 }
} else {
 if(i24<16914.5){
  if(i10<906.369140625){
   s1+=7.0;
  } else {
   s0+=3.0;
  }
 } else {
  if(i18<440.623046875){
   s0+=1.0;
   s1+=38.0;
  } else {
   s1+=692.0;
  }
 }
}
if(i10<173.07958984375){
 if(i6<25595.88671875){
  if(i4<20676.931640625){
   s0+=1149.0;
   s1+=2.0;
  } else {
   s0+=4.0;
   s1+=4.0;
  }
 } else {
  s1+=12.0;
 }
} else {
 if(i20<10778.259765625){
  if(i7<14168.33984375){
   s0+=8.0;
  } else {
   s1+=13.0;
  }
 } else {
  if(i0<13095.6455078125){
   s0+=2.0;
  } else {
   s0+=1.0;
   s1+=758.0;
  }
 }
}
if(i13<294.052490234375){
 if(i5<25247.765625){
  if(i2<4825.7861328125){
   s0+=1136.0;
   s1+=4.0;
  } else {
   s0+=1.0;
   s1+=5.0;
  }
 } else {
  s1+=17.0;
 }
} else {
 if(i10<349.48583984375){
  if(i6<28193.498046875){
   s0+=5.0;
   s1+=1.0;
  } else {
   s1+=5.0;
  }
 } else {
  if(i5<16162.72265625){
   s0+=4.0;
   s1+=14.0;
  } else {
   s1+=761.0;
  }
 }
}
if(i17<464.15673828125){
 if(i0<18402.638671875){
  if(i1<535.21435546875){
   s0+=1207.0;
   s1+=1.0;
  } else {
   s1+=2.0;
  }
 } else {
  if(i3<78784.6875){
   s1+=24.0;
  } else {
   s0+=3.0;
  }
 }
} else {
 if(i10<350.2734375){
  if(i21<1970.0927734375){
   s0+=11.0;
  } else {
   s1+=11.0;
  }
 } else {
  if(i24<16997.5){
   s0+=3.0;
   s1+=9.0;
  } else {
   s0+=1.0;
   s1+=681.0;
  }
 }
}
if(i4<17951.0859375){
 if(i24<15491.5){
  if(i13<478.27587890625){
   s0+=1201.0;
  } else {
   s0+=6.0;
   s1+=1.0;
  }
 } else {
  s1+=13.0;
 }
} else {
 if(i9<572.783203125){
  if(i3<60343.78125){
   s1+=24.0;
  } else {
   s0+=16.0;
  }
 } else {
  if(i3<106959.328125){
   s1+=686.0;
  } else {
   s0+=1.0;
   s1+=5.0;
  }
 }
}
if(i17<464.15673828125){
 if(i22<17679.4765625){
  if(i1<436.38818359375){
   s0+=1150.0;
   s1+=5.0;
  } else {
   s0+=1.0;
   s1+=10.0;
  }
 } else {
  s1+=10.0;
 }
} else {
 if(i2<-8732.03125){
  if(i24<16466.5){
   s0+=14.0;
   s1+=1.0;
  } else {
   s0+=1.0;
   s1+=67.0;
  }
 } else {
  s1+=694.0;
 }
}
if(i4<18245.759765625){
 if(i12<1055.544677734375){
  if(i10<650.46337890625){
   s0+=1180.0;
   s1+=1.0;
  } else {
   s1+=1.0;
  }
 } else {
  if(i17<813.90283203125){
   s1+=18.0;
  } else {
   s0+=1.0;
  }
 }
} else {
 if(i24<13942.5){
  s0+=11.0;
 } else {
  if(i3<100190.28125){
   s1+=727.0;
  } else {
   s0+=2.0;
   s1+=12.0;
  }
 }
}
 float max_s=s0;
 int cls=1;
 if (max_s < s1) {
  max_s = s1;
  cls=2;
 }
 WRITE_IMAGE (out, POS_out_INSTANCE(x,y,z,0), cls);
}
