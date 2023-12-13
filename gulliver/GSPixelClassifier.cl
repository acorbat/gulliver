/*
OpenCL RandomForestClassifier
classifier_class_name = PixelClassifier
feature_specification = gaussian_blur=1 difference_of_gaussian=1 laplace_box_of_gaussian_blur=1 sobel_of_gaussian_blur=1 gaussian_blur=25 difference_of_gaussian=25 laplace_box_of_gaussian_blur=25 sobel_of_gaussian_blur=25 sobel_of_gaussian_blur=15 laplace_box_of_gaussian_blur=15 difference_of_gaussian=15 gaussian_blur=15 gaussian_blur=3 difference_of_gaussian=3 laplace_box_of_gaussian_blur=3 sobel_of_gaussian_blur=3 original
num_ground_truth_dimensions = 2
num_classes = 2
num_features = 17
max_depth = 3
num_trees = 100
feature_importances = 0.05016816017089393,0.0,0.0,0.0002566628909289911,0.05436667670543859,0.24995352194385767,0.22342506197941997,0.024231146389689946,0.019748566516116776,0.02503351242216137,0.017850800956081242,0.15049094358060594,0.10725071373952798,0.00030819419559179756,0.00018441725975060333,0.0007832832762186815,0.07594833797371653
apoc_version = 0.12.0
*/
__kernel void predict (IMAGE_in0_TYPE in0, IMAGE_in1_TYPE in1, IMAGE_in2_TYPE in2, IMAGE_in3_TYPE in3, IMAGE_in4_TYPE in4, IMAGE_in5_TYPE in5, IMAGE_in6_TYPE in6, IMAGE_in7_TYPE in7, IMAGE_in8_TYPE in8, IMAGE_in9_TYPE in9, IMAGE_in10_TYPE in10, IMAGE_in11_TYPE in11, IMAGE_in12_TYPE in12, IMAGE_in13_TYPE in13, IMAGE_in14_TYPE in14, IMAGE_in15_TYPE in15, IMAGE_in16_TYPE in16, IMAGE_out_TYPE out) {
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
 float s0=0;
 float s1=0;
if(i6<0.8546142578125){
 s0+=13212.0;
} else {
 if(i6<6.0220947265625){
  if(i0<5263.2451171875){
   s0+=72.0;
   s1+=3473.0;
  } else {
   s0+=30.0;
   s1+=2.0;
  }
 } else {
  s0+=536.0;
 }
}
if(i4<2003.29296875){
 if(i8<228.2986297607422){
  s0+=12210.0;
 } else {
  if(i5<38.918701171875){
   s0+=692.0;
  } else {
   s1+=85.0;
  }
 }
} else {
 if(i8<418.35504150390625){
  if(i5<284.97607421875){
   s0+=339.0;
   s1+=3348.0;
  } else {
   s0+=347.0;
  }
 } else {
  if(i0<3725.03857421875){
   s0+=18.0;
   s1+=4.0;
  } else {
   s0+=281.0;
   s1+=1.0;
  }
 }
}
if(i7<88.45755767822266){
 if(i4<2453.80419921875){
  if(i6<1.6708984375){
   s0+=12523.0;
  } else {
   s1+=55.0;
  }
 } else {
  if(i0<5496.9208984375){
   s1+=683.0;
  } else {
   s0+=134.0;
  }
 }
} else {
 if(i8<399.340087890625){
  if(i10<298.7498779296875){
   s0+=18.0;
   s1+=2667.0;
  } else {
   s0+=218.0;
   s1+=9.0;
  }
 } else {
  if(i7<273.7901611328125){
   s0+=185.0;
   s1+=37.0;
  } else {
   s0+=793.0;
   s1+=3.0;
  }
 }
}
if(i5<36.00360107421875){
 s0+=13156.0;
} else {
 if(i6<6.0220947265625){
  if(i15<2395.75830078125){
   s0+=70.0;
   s1+=3528.0;
  } else {
   s0+=35.0;
   s1+=6.0;
  }
 } else {
  s0+=530.0;
 }
}
if(i6<0.854736328125){
 s0+=13189.0;
} else {
 if(i11<3934.58154296875){
  if(i6<3.0787353515625){
   s0+=9.0;
   s1+=2473.0;
  } else {
   s0+=106.0;
   s1+=1027.0;
  }
 } else {
  s0+=521.0;
 }
}
if(i5<36.02569580078125){
 s0+=13243.0;
} else {
 if(i12<4795.33203125){
  if(i4<3441.89990234375){
   s0+=110.0;
   s1+=3291.0;
  } else {
   s0+=14.0;
  }
 } else {
  if(i11<3929.255859375){
   s0+=57.0;
   s1+=102.0;
  } else {
   s0+=508.0;
  }
 }
}
if(i7<88.0430908203125){
 if(i15<391.7729187011719){
  if(i8<65.75332641601562){
   s0+=8263.0;
   s1+=7.0;
  } else {
   s0+=1329.0;
   s1+=221.0;
  }
 } else {
  if(i5<35.56378173828125){
   s0+=2967.0;
  } else {
   s0+=95.0;
   s1+=544.0;
  }
 }
} else {
 if(i0<1943.4439697265625){
  if(i7<226.5168914794922){
   s0+=6.0;
   s1+=8.0;
  } else {
   s0+=683.0;
  }
 } else {
  if(i4<3396.416015625){
   s0+=148.0;
   s1+=2696.0;
  } else {
   s0+=358.0;
  }
 }
}
if(i5<35.56378173828125){
 s0+=13226.0;
} else {
 if(i6<6.0455322265625){
  if(i8<479.6573486328125){
   s1+=3411.0;
  } else {
   s0+=102.0;
  }
 } else {
  s0+=586.0;
 }
}
if(i5<35.56378173828125){
 s0+=13283.0;
} else {
 if(i6<6.0203857421875){
  if(i8<479.160400390625){
   s1+=3436.0;
  } else {
   s0+=109.0;
  }
 } else {
  s0+=497.0;
 }
}
if(i11<2191.5048828125){
 if(i11<2089.0546875){
  if(i15<1122.77392578125){
   s0+=12658.0;
   s1+=2.0;
  } else {
   s0+=175.0;
   s1+=17.0;
  }
 } else {
  if(i9<0.0634765625){
   s0+=230.0;
  } else {
   s1+=88.0;
  }
 }
} else {
 if(i10<280.8623046875){
  if(i5<251.7432861328125){
   s0+=131.0;
   s1+=3404.0;
  } else {
   s0+=98.0;
  }
 } else {
  if(i9<20.751953125){
   s0+=54.0;
   s1+=32.0;
  } else {
   s0+=435.0;
   s1+=1.0;
  }
 }
}
if(i16<2568.0){
 if(i10<57.94915771484375){
  if(i11<2215.0185546875){
   s0+=13015.0;
   s1+=28.0;
  } else {
   s0+=4.0;
   s1+=18.0;
  }
 } else {
  if(i5<47.88714599609375){
   s0+=34.0;
  } else {
   s1+=50.0;
  }
 }
} else {
 if(i5<251.7432861328125){
  if(i6<0.832275390625){
   s0+=132.0;
  } else {
   s0+=112.0;
   s1+=3377.0;
  }
 } else {
  s0+=555.0;
 }
}
if(i12<2491.174072265625){
 if(i6<1.02423095703125){
  s0+=12971.0;
 } else {
  s1+=64.0;
 }
} else {
 if(i11<3924.7763671875){
  if(i11<2065.939453125){
   s0+=78.0;
  } else {
   s0+=123.0;
   s1+=3469.0;
  }
 } else {
  s0+=620.0;
 }
}
if(i5<35.56378173828125){
 s0+=13158.0;
} else {
 if(i6<6.0220947265625){
  if(i3<4591.6630859375){
   s0+=75.0;
   s1+=3519.0;
  } else {
   s0+=23.0;
   s1+=5.0;
  }
 } else {
  s0+=545.0;
 }
}
if(i12<2491.021484375){
 if(i11<2218.00390625){
  if(i11<2123.86767578125){
   s0+=13013.0;
   s1+=13.0;
  } else {
   s0+=175.0;
   s1+=21.0;
  }
 } else {
  s1+=33.0;
 }
} else {
 if(i5<251.7432861328125){
  if(i10<46.0594482421875){
   s0+=94.0;
   s1+=144.0;
  } else {
   s0+=109.0;
   s1+=3198.0;
  }
 } else {
  s0+=525.0;
 }
}
if(i6<0.8548583984375){
 s0+=13153.0;
} else {
 if(i11<3924.25830078125){
  if(i13<113.2115478515625){
   s0+=67.0;
   s1+=3397.0;
  } else {
   s0+=66.0;
   s1+=102.0;
  }
 } else {
  s0+=540.0;
 }
}
if(i6<0.8546142578125){
 s0+=13175.0;
} else {
 if(i11<3924.296630859375){
  if(i7<286.1624450683594){
   s0+=5.0;
   s1+=3496.0;
  } else {
   s0+=112.0;
  }
 } else {
  s0+=537.0;
 }
}
if(i11<2193.59423828125){
 if(i6<0.9393310546875){
  s0+=13041.0;
 } else {
  s1+=115.0;
 }
} else {
 if(i9<17.888916015625){
  if(i11<3924.7763671875){
   s0+=144.0;
   s1+=3399.0;
  } else {
   s0+=65.0;
  }
 } else {
  if(i4<3267.451416015625){
   s1+=52.0;
  } else {
   s0+=509.0;
  }
 }
}
if(i12<2491.174072265625){
 if(i11<2214.735107421875){
  if(i12<2382.64453125){
   s0+=12926.0;
   s1+=14.0;
  } else {
   s0+=89.0;
   s1+=21.0;
  }
 } else {
  s1+=28.0;
 }
} else {
 if(i4<3441.63818359375){
  if(i8<434.30889892578125){
   s0+=87.0;
   s1+=3485.0;
  } else {
   s0+=168.0;
  }
 } else {
  s0+=507.0;
 }
}
if(i5<36.00299072265625){
 s0+=13228.0;
} else {
 if(i12<5017.5166015625){
  if(i8<431.5855712890625){
   s0+=27.0;
   s1+=3424.0;
  } else {
   s0+=209.0;
   s1+=2.0;
  }
 } else {
  if(i4<3132.497314453125){
   s1+=8.0;
  } else {
   s0+=427.0;
  }
 }
}
if(i9<3.83380126953125){
 if(i6<0.8548583984375){
  s0+=13294.0;
 } else {
  if(i15<2436.62353515625){
   s1+=285.0;
  } else {
   s0+=2.0;
  }
 }
} else {
 if(i6<6.0220947265625){
  if(i8<458.0500183105469){
   s0+=12.0;
   s1+=3065.0;
  } else {
   s0+=130.0;
  }
 } else {
  s0+=537.0;
 }
}
if(i5<35.54168701171875){
 s0+=13103.0;
} else {
 if(i6<6.0220947265625){
  if(i8<469.1168212890625){
   s1+=3581.0;
  } else {
   s0+=111.0;
  }
 } else {
  s0+=530.0;
 }
}
if(i10<50.27880859375){
 if(i5<36.00360107421875){
  s0+=13059.0;
 } else {
  if(i13<-83.6055908203125){
   s0+=2.0;
   s1+=2.0;
  } else {
   s1+=200.0;
  }
 }
} else {
 if(i11<3924.296630859375){
  if(i5<38.87164306640625){
   s0+=201.0;
  } else {
   s0+=109.0;
   s1+=3245.0;
  }
 } else {
  s0+=507.0;
 }
}
if(i6<0.8548583984375){
 s0+=13181.0;
} else {
 if(i11<3924.7763671875){
  if(i8<459.5782470703125){
   s1+=3521.0;
  } else {
   s0+=110.0;
  }
 } else {
  s0+=513.0;
 }
}
if(i6<0.868896484375){
 s0+=13171.0;
} else {
 if(i5<251.7432861328125){
  if(i8<468.4366455078125){
   s1+=3525.0;
  } else {
   s0+=88.0;
  }
 } else {
  s0+=541.0;
 }
}
if(i5<36.00360107421875){
 s0+=13239.0;
} else {
 if(i4<3436.63671875){
  if(i6<6.0220947265625){
   s0+=113.0;
   s1+=3427.0;
  } else {
   s0+=49.0;
  }
 } else {
  s0+=497.0;
 }
}
if(i11<2191.275390625){
 if(i6<0.939208984375){
  s0+=13143.0;
 } else {
  s1+=96.0;
 }
} else {
 if(i6<6.0220947265625){
  if(i6<0.0955810546875){
   s0+=26.0;
  } else {
   s0+=99.0;
   s1+=3400.0;
  }
 } else {
  s0+=561.0;
 }
}
if(i11<2193.59423828125){
 if(i0<2569.535888671875){
  if(i9<3.85198974609375){
   s0+=12978.0;
   s1+=20.0;
  } else {
   s0+=49.0;
   s1+=14.0;
  }
 } else {
  if(i7<84.6875228881836){
   s0+=106.0;
  } else {
   s1+=64.0;
  }
 }
} else {
 if(i6<6.0220947265625){
  if(i7<276.44769287109375){
   s0+=24.0;
   s1+=3373.0;
  } else {
   s0+=121.0;
   s1+=6.0;
  }
 } else {
  s0+=570.0;
 }
}
if(i11<2193.6943359375){
 if(i6<0.952392578125){
  s0+=13160.0;
 } else {
  s1+=113.0;
 }
} else {
 if(i12<4757.57666015625){
  if(i7<274.2637939453125){
   s0+=40.0;
   s1+=3271.0;
  } else {
   s0+=90.0;
   s1+=3.0;
  }
 } else {
  if(i6<6.0250244140625){
   s0+=24.0;
   s1+=93.0;
  } else {
   s0+=531.0;
  }
 }
}
if(i11<2162.734375){
 if(i3<1483.17919921875){
  if(i16<2754.5){
   s0+=12619.0;
   s1+=13.0;
  } else {
   s0+=10.0;
   s1+=32.0;
  }
 } else {
  if(i6<0.967041015625){
   s0+=368.0;
  } else {
   s1+=40.0;
  }
 }
} else {
 if(i4<3436.63671875){
  if(i10<1.866943359375){
   s0+=71.0;
  } else {
   s0+=158.0;
   s1+=3466.0;
  }
 } else {
  s0+=548.0;
 }
}
if(i11<2191.607421875){
 if(i11<2123.86767578125){
  if(i9<2.77239990234375){
   s0+=12600.0;
   s1+=6.0;
  } else {
   s0+=370.0;
   s1+=20.0;
  }
 } else {
  if(i7<112.25374603271484){
   s0+=169.0;
  } else {
   s1+=78.0;
  }
 }
} else {
 if(i0<5159.826171875){
  if(i5<253.6978759765625){
   s0+=97.0;
   s1+=3363.0;
  } else {
   s0+=114.0;
  }
 } else {
  if(i4<3062.092529296875){
   s1+=6.0;
  } else {
   s0+=495.0;
   s1+=7.0;
  }
 }
}
if(i5<35.56378173828125){
 s0+=13176.0;
} else {
 if(i11<3924.7763671875){
  if(i0<5226.189453125){
   s0+=79.0;
   s1+=3468.0;
  } else {
   s0+=40.0;
   s1+=5.0;
  }
 } else {
  s0+=557.0;
 }
}
if(i12<2488.596923828125){
 if(i12<2359.591796875){
  if(i10<42.44537353515625){
   s0+=12636.0;
   s1+=7.0;
  } else {
   s0+=296.0;
   s1+=23.0;
  }
 } else {
  if(i6<0.9853515625){
   s0+=142.0;
  } else {
   s1+=53.0;
  }
 }
} else {
 if(i16<5150.5){
  if(i8<431.5855712890625){
   s0+=131.0;
   s1+=3373.0;
  } else {
   s0+=139.0;
   s1+=1.0;
  }
 } else {
  if(i8<110.41909790039062){
   s0+=14.0;
   s1+=33.0;
  } else {
   s0+=471.0;
   s1+=6.0;
  }
 }
}
if(i6<0.869140625){
 s0+=13205.0;
} else {
 if(i6<6.0220947265625){
  if(i8<469.83123779296875){
   s1+=3460.0;
  } else {
   s0+=104.0;
  }
 } else {
  s0+=556.0;
 }
}
if(i11<2191.630859375){
 if(i10<57.5579833984375){
  if(i16<2574.5){
   s0+=12989.0;
   s1+=17.0;
  } else {
   s0+=138.0;
   s1+=25.0;
  }
 } else {
  if(i4<1927.678955078125){
   s0+=50.0;
  } else {
   s1+=71.0;
  }
 }
} else {
 if(i11<3922.181640625){
  if(i8<458.86383056640625){
   s0+=16.0;
   s1+=3316.0;
  } else {
   s0+=128.0;
  }
 } else {
  if(i9<14.3280029296875){
   s0+=17.0;
   s1+=1.0;
  } else {
   s0+=557.0;
  }
 }
}
if(i11<2192.05029296875){
 if(i9<3.83404541015625){
  if(i0<2430.68603515625){
   s0+=12849.0;
   s1+=11.0;
  } else {
   s0+=259.0;
   s1+=31.0;
  }
 } else {
  if(i12<2399.31494140625){
   s0+=40.0;
   s1+=3.0;
  } else {
   s1+=52.0;
  }
 }
} else {
 if(i6<6.0203857421875){
  if(i0<5217.35498046875){
   s0+=86.0;
   s1+=3381.0;
  } else {
   s0+=25.0;
   s1+=4.0;
  }
 } else {
  s0+=584.0;
 }
}
if(i5<36.02569580078125){
 s0+=13170.0;
} else {
 if(i5<251.7432861328125){
  if(i8<479.160400390625){
   s1+=3435.0;
  } else {
   s0+=108.0;
  }
 } else {
  s0+=612.0;
 }
}
if(i4<2002.106201171875){
 if(i9<3.8609619140625){
  if(i12<2471.4541015625){
   s0+=12829.0;
   s1+=20.0;
  } else {
   s0+=82.0;
   s1+=17.0;
  }
 } else {
  if(i4<1926.501708984375){
   s0+=39.0;
  } else {
   s1+=45.0;
  }
 }
} else {
 if(i0<5158.26318359375){
  if(i12<2206.39697265625){
   s0+=351.0;
   s1+=4.0;
  } else {
   s0+=177.0;
   s1+=3260.0;
  }
 } else {
  if(i4<3093.45849609375){
   s0+=1.0;
   s1+=6.0;
  } else {
   s0+=488.0;
   s1+=6.0;
  }
 }
}
if(i6<0.8548583984375){
 s0+=13194.0;
} else {
 if(i5<252.6951904296875){
  if(i7<280.36151123046875){
   s1+=3457.0;
  } else {
   s0+=115.0;
   s1+=3.0;
  }
 } else {
  s0+=556.0;
 }
}
if(i12<2491.9033203125){
 if(i15<1060.88818359375){
  if(i10<60.5694580078125){
   s0+=12862.0;
   s1+=5.0;
  } else {
   s1+=2.0;
  }
 } else {
  if(i12<1992.92626953125){
   s0+=166.0;
  } else {
   s0+=44.0;
   s1+=64.0;
  }
 }
} else {
 if(i5<253.6451416015625){
  if(i7<286.1624450683594){
   s0+=67.0;
   s1+=3439.0;
  } else {
   s0+=112.0;
  }
 } else {
  s0+=564.0;
 }
}
if(i7<88.04802703857422){
 if(i6<0.854736328125){
  s0+=12539.0;
 } else {
  if(i4<3715.56298828125){
   s1+=805.0;
  } else {
   s0+=139.0;
  }
 }
} else {
 if(i5<38.87164306640625){
  s0+=633.0;
 } else {
  if(i16<4885.0){
   s0+=122.0;
   s1+=2566.0;
  } else {
   s0+=418.0;
   s1+=103.0;
  }
 }
}
if(i5<36.00360107421875){
 s0+=13179.0;
} else {
 if(i4<3436.63671875){
  if(i7<275.9449462890625){
   s0+=12.0;
   s1+=3485.0;
  } else {
   s0+=158.0;
   s1+=3.0;
  }
 } else {
  s0+=488.0;
 }
}
if(i0<2566.53857421875){
 if(i11<2218.00390625){
  if(i10<58.64654541015625){
   s0+=13041.0;
   s1+=27.0;
  } else {
   s0+=34.0;
   s1+=14.0;
  }
 } else {
  if(i5<101.005126953125){
   s1+=49.0;
  } else {
   s0+=8.0;
  }
 }
} else {
 if(i0<5141.0810546875){
  if(i11<2083.915771484375){
   s0+=93.0;
   s1+=1.0;
  } else {
   s0+=153.0;
   s1+=3393.0;
  }
 } else {
  if(i11<3924.25830078125){
   s0+=48.0;
   s1+=17.0;
  } else {
   s0+=447.0;
  }
 }
}
if(i12<2463.6123046875){
 if(i10<42.4344482421875){
  if(i6<1.05975341796875){
   s0+=12774.0;
  } else {
   s1+=14.0;
  }
 } else {
  if(i6<1.0118408203125){
   s0+=348.0;
  } else {
   s1+=66.0;
  }
 }
} else {
 if(i11<3934.101806640625){
  if(i13<107.2763671875){
   s0+=114.0;
   s1+=3277.0;
  } else {
   s0+=99.0;
   s1+=97.0;
  }
 } else {
  s0+=536.0;
 }
}
if(i5<36.00360107421875){
 s0+=13149.0;
} else {
 if(i6<6.0220947265625){
  if(i3<4048.831298828125){
   s0+=75.0;
   s1+=3535.0;
  } else {
   s0+=23.0;
   s1+=7.0;
  }
 } else {
  s0+=536.0;
 }
}
if(i5<35.56378173828125){
 s0+=13155.0;
} else {
 if(i5<253.60791015625){
  if(i7<282.0853271484375){
   s0+=2.0;
   s1+=3492.0;
  } else {
   s0+=86.0;
   s1+=1.0;
  }
 } else {
  s0+=589.0;
 }
}
if(i4<2002.106201171875){
 if(i11<2085.61865234375){
  if(i5<45.1793212890625){
   s0+=12891.0;
  } else {
   s1+=9.0;
  }
 } else {
  s1+=72.0;
 }
} else {
 if(i5<251.6790771484375){
  if(i7<81.68904113769531){
   s0+=364.0;
   s1+=686.0;
  } else {
   s0+=97.0;
   s1+=2679.0;
  }
 } else {
  s0+=527.0;
 }
}
if(i12<2491.021484375){
 if(i3<1539.6893310546875){
  if(i0<2610.836669921875){
   s0+=12824.0;
   s1+=10.0;
  } else {
   s0+=3.0;
   s1+=4.0;
  }
 } else {
  if(i6<0.9967041015625){
   s0+=287.0;
  } else {
   s1+=60.0;
  }
 }
} else {
 if(i6<6.0220947265625){
  if(i6<0.69287109375){
   s0+=84.0;
  } else {
   s0+=115.0;
   s1+=3397.0;
  }
 } else {
  s0+=541.0;
 }
}
if(i12<2478.9873046875){
 if(i6<1.00189208984375){
  s0+=13145.0;
 } else {
  s1+=70.0;
 }
} else {
 if(i6<6.0316162109375){
  if(i7<286.3001708984375){
   s0+=83.0;
   s1+=3390.0;
  } else {
   s0+=82.0;
  }
 } else {
  s0+=555.0;
 }
}
if(i16<2663.0){
 if(i16<2428.5){
  if(i5<45.1793212890625){
   s0+=12848.0;
  } else {
   s1+=45.0;
  }
 } else {
  if(i9<4.010009765625){
   s0+=306.0;
   s1+=43.0;
  } else {
   s1+=66.0;
  }
 }
} else {
 if(i0<5212.67431640625){
  if(i4<3436.63671875){
   s0+=208.0;
   s1+=3285.0;
  } else {
   s0+=82.0;
  }
 } else {
  if(i4<3062.092529296875){
   s1+=2.0;
  } else {
   s0+=438.0;
   s1+=2.0;
  }
 }
}
if(i16<2566.5){
 if(i5<44.0072021484375){
  s0+=12997.0;
 } else {
  if(i8<552.4417114257812){
   s1+=108.0;
  } else {
   s0+=5.0;
  }
 }
} else {
 if(i6<6.0220947265625){
  if(i5<35.04278564453125){
   s0+=171.0;
  } else {
   s0+=105.0;
   s1+=3335.0;
  }
 } else {
  s0+=604.0;
 }
}
if(i6<0.8634033203125){
 s0+=13190.0;
} else {
 if(i5<252.657958984375){
  if(i8<469.83123779296875){
   s1+=3456.0;
  } else {
   s0+=104.0;
  }
 } else {
  s0+=575.0;
 }
}
if(i5<36.02569580078125){
 s0+=13131.0;
} else {
 if(i6<6.0316162109375){
  if(i0<5171.35302734375){
   s0+=66.0;
   s1+=3515.0;
  } else {
   s0+=46.0;
   s1+=9.0;
  }
 } else {
  s0+=558.0;
 }
}
if(i6<0.8690185546875){
 s0+=13193.0;
} else {
 if(i4<3436.63671875){
  if(i15<2397.6328125){
   s0+=136.0;
   s1+=3469.0;
  } else {
   s0+=44.0;
   s1+=8.0;
  }
 } else {
  s0+=475.0;
 }
}
if(i6<0.8548583984375){
 s0+=13231.0;
} else {
 if(i6<6.0220947265625){
  if(i3<4298.6767578125){
   s0+=71.0;
   s1+=3469.0;
  } else {
   s0+=23.0;
   s1+=4.0;
  }
 } else {
  s0+=527.0;
 }
}
if(i5<35.56378173828125){
 s0+=13196.0;
} else {
 if(i0<5171.35302734375){
  if(i8<433.594482421875){
   s0+=35.0;
   s1+=3455.0;
  } else {
   s0+=159.0;
  }
 } else {
  if(i15<141.58059692382812){
   s0+=7.0;
   s1+=5.0;
  } else {
   s0+=458.0;
   s1+=10.0;
  }
 }
}
if(i9<3.375732421875){
 if(i4<2484.46533203125){
  if(i11<2214.15625){
   s0+=13049.0;
   s1+=17.0;
  } else {
   s1+=56.0;
  }
 } else {
  if(i15<1897.882568359375){
   s1+=146.0;
  } else {
   s0+=1.0;
  }
 }
} else {
 if(i11<3924.7763671875){
  if(i0<2489.6259765625){
   s0+=191.0;
   s1+=42.0;
  } else {
   s0+=125.0;
   s1+=3175.0;
  }
 } else {
  s0+=523.0;
 }
}
if(i5<35.56378173828125){
 s0+=13117.0;
} else {
 if(i4<3436.63671875){
  if(i16<5155.0){
   s0+=113.0;
   s1+=3457.0;
  } else {
   s0+=62.0;
   s1+=38.0;
  }
 } else {
  s0+=538.0;
 }
}
if(i12<2491.021484375){
 if(i5<42.187255859375){
  s0+=13139.0;
 } else {
  s1+=68.0;
 }
} else {
 if(i0<5158.26318359375){
  if(i7<272.40777587890625){
   s0+=142.0;
   s1+=3375.0;
  } else {
   s0+=107.0;
   s1+=10.0;
  }
 } else {
  if(i4<3101.67431640625){
   s0+=2.0;
   s1+=11.0;
  } else {
   s0+=465.0;
   s1+=6.0;
  }
 }
}
if(i11<2167.68115234375){
 if(i10<71.5018310546875){
  if(i5<41.47418212890625){
   s0+=13128.0;
  } else {
   s1+=61.0;
  }
 } else {
  if(i3<1807.9517822265625){
   s1+=16.0;
  } else {
   s0+=1.0;
  }
 }
} else {
 if(i5<253.60791015625){
  if(i10<1.0545654296875){
   s0+=60.0;
  } else {
   s0+=104.0;
   s1+=3429.0;
  }
 } else {
  s0+=526.0;
 }
}
if(i0<2591.53759765625){
 if(i11<2215.0185546875){
  if(i15<911.7229614257812){
   s0+=12871.0;
   s1+=1.0;
  } else {
   s0+=284.0;
   s1+=47.0;
  }
 } else {
  if(i9<0.129150390625){
   s0+=1.0;
  } else {
   s0+=1.0;
   s1+=36.0;
  }
 }
} else {
 if(i6<6.0406494140625){
  if(i10<44.5042724609375){
   s0+=91.0;
   s1+=161.0;
  } else {
   s0+=106.0;
   s1+=3175.0;
  }
 } else {
  s0+=551.0;
 }
}
if(i12<2496.743408203125){
 if(i5<42.2093505859375){
  s0+=13114.0;
 } else {
  s1+=64.0;
 }
} else {
 if(i16<5188.0){
  if(i5<253.66064453125){
   s0+=165.0;
   s1+=3333.0;
  } else {
   s0+=126.0;
  }
 } else {
  if(i3<654.203857421875){
   s0+=37.0;
   s1+=12.0;
  } else {
   s0+=460.0;
   s1+=14.0;
  }
 }
}
if(i11<2192.28857421875){
 if(i0<2592.25){
  if(i5<42.5849609375){
   s0+=13110.0;
  } else {
   s1+=33.0;
  }
 } else {
  if(i8<171.6468048095703){
   s0+=86.0;
  } else {
   s1+=67.0;
  }
 }
} else {
 if(i6<6.0220947265625){
  if(i12<2211.041015625){
   s0+=22.0;
   s1+=8.0;
  } else {
   s0+=103.0;
   s1+=3361.0;
  }
 } else {
  s0+=535.0;
 }
}
if(i12<2463.94140625){
 if(i6<1.01580810546875){
  s0+=13102.0;
 } else {
  s1+=65.0;
 }
} else {
 if(i6<6.0406494140625){
  if(i8<479.160400390625){
   s0+=95.0;
   s1+=3416.0;
  } else {
   s0+=94.0;
  }
 } else {
  s0+=553.0;
 }
}
if(i5<36.02569580078125){
 s0+=13211.0;
} else {
 if(i0<5212.67431640625){
  if(i8<433.594482421875){
   s0+=25.0;
   s1+=3465.0;
  } else {
   s0+=173.0;
  }
 } else {
  if(i4<3054.667236328125){
   s1+=4.0;
  } else {
   s0+=446.0;
   s1+=1.0;
  }
 }
}
if(i7<88.45292663574219){
 if(i9<3.97930908203125){
  if(i0<2666.6728515625){
   s0+=12418.0;
  } else {
   s0+=68.0;
   s1+=220.0;
  }
 } else {
  if(i11<4550.94091796875){
   s1+=534.0;
  } else {
   s0+=120.0;
  }
 }
} else {
 if(i8<399.40777587890625){
  if(i11<4103.9296875){
   s0+=6.0;
   s1+=2689.0;
  } else {
   s0+=205.0;
  }
 } else {
  if(i10<77.2938232421875){
   s0+=708.0;
  } else {
   s0+=318.0;
   s1+=39.0;
  }
 }
}
if(i6<0.874267578125){
 s0+=13076.0;
} else {
 if(i5<252.657958984375){
  if(i16<5033.5){
   s0+=52.0;
   s1+=3530.0;
  } else {
   s0+=55.0;
   s1+=56.0;
  }
 } else {
  s0+=556.0;
 }
}
if(i11<2163.40576171875){
 if(i12<2491.021484375){
  if(i5<44.4158935546875){
   s0+=13133.0;
  } else {
   s1+=23.0;
  }
 } else {
  if(i8<175.87274169921875){
   s0+=69.0;
  } else {
   s1+=26.0;
  }
 }
} else {
 if(i6<6.0220947265625){
  if(i8<469.83123779296875){
   s0+=75.0;
   s1+=3382.0;
  } else {
   s0+=106.0;
  }
 } else {
  s0+=511.0;
 }
}
if(i16<2609.5){
 if(i10<58.47271728515625){
  if(i5<36.33673095703125){
   s0+=13110.0;
  } else {
   s0+=2.0;
   s1+=59.0;
  }
 } else {
  if(i0<2017.999267578125){
   s0+=19.0;
  } else {
   s0+=11.0;
   s1+=66.0;
  }
 }
} else {
 if(i5<251.7432861328125){
  if(i6<0.8248291015625){
   s0+=118.0;
  } else {
   s0+=82.0;
   s1+=3309.0;
  }
 } else {
  s0+=549.0;
 }
}
if(i16<2559.5){
 if(i9<3.99468994140625){
  if(i15<1075.406982421875){
   s0+=12862.0;
   s1+=5.0;
  } else {
   s0+=196.0;
   s1+=47.0;
  }
 } else {
  if(i4<1924.70068359375){
   s0+=11.0;
  } else {
   s1+=43.0;
  }
 }
} else {
 if(i5<251.7432861328125){
  if(i12<2463.94140625){
   s0+=83.0;
   s1+=9.0;
  } else {
   s0+=166.0;
   s1+=3347.0;
  }
 } else {
  s0+=556.0;
 }
}
if(i6<0.869140625){
 s0+=13211.0;
} else {
 if(i0<5226.189453125){
  if(i11<3934.58154296875){
   s0+=76.0;
   s1+=3494.0;
  } else {
   s0+=107.0;
  }
 } else {
  if(i11<3768.5048828125){
   s0+=21.0;
   s1+=4.0;
  } else {
   s0+=411.0;
   s1+=1.0;
  }
 }
}
if(i5<36.35882568359375){
 s0+=13208.0;
} else {
 if(i11<3924.296630859375){
  if(i16<5282.5){
   s0+=95.0;
   s1+=3431.0;
  } else {
   s0+=41.0;
   s1+=8.0;
  }
 } else {
  s0+=542.0;
 }
}
if(i0<2591.499267578125){
 if(i5<42.5849609375){
  s0+=13202.0;
 } else {
  s1+=104.0;
 }
} else {
 if(i0<5158.26318359375){
  if(i8<431.5855712890625){
   s0+=110.0;
   s1+=3278.0;
  } else {
   s0+=160.0;
  }
 } else {
  if(i15<107.38658142089844){
   s1+=2.0;
  } else {
   s0+=463.0;
   s1+=6.0;
  }
 }
}
if(i11<2162.984375){
 if(i6<0.939453125){
  s0+=13116.0;
 } else {
  s1+=59.0;
 }
} else {
 if(i5<251.665283203125){
  if(i6<0.1025390625){
   s0+=80.0;
  } else {
   s0+=111.0;
   s1+=3415.0;
  }
 } else {
  s0+=544.0;
 }
}
if(i6<0.8548583984375){
 s0+=13182.0;
} else {
 if(i12<4848.5986328125){
  if(i8<431.5855712890625){
   s0+=5.0;
   s1+=3451.0;
  } else {
   s0+=107.0;
   s1+=1.0;
  }
 } else {
  if(i5<251.2340087890625){
   s0+=21.0;
   s1+=57.0;
  } else {
   s0+=501.0;
  }
 }
}
if(i5<36.02569580078125){
 s0+=13190.0;
} else {
 if(i6<6.0299072265625){
  if(i16<5030.0){
   s0+=68.0;
   s1+=3376.0;
  } else {
   s0+=60.0;
   s1+=64.0;
  }
 } else {
  s0+=567.0;
 }
}
if(i5<35.54107666015625){
 s0+=13198.0;
} else {
 if(i16<5159.0){
  if(i7<271.64794921875){
   s0+=58.0;
   s1+=3436.0;
  } else {
   s0+=102.0;
   s1+=6.0;
  }
 } else {
  if(i0<5260.728515625){
   s0+=33.0;
   s1+=36.0;
  } else {
   s0+=455.0;
   s1+=1.0;
  }
 }
}
if(i8<99.52375793457031){
 if(i11<2351.0986328125){
  s0+=11553.0;
 } else {
  if(i10<389.2181396484375){
   s1+=653.0;
  } else {
   s0+=24.0;
  }
 }
} else {
 if(i12<2184.2939453125){
  if(i16<1619.5){
   s0+=1230.0;
  } else {
   s0+=310.0;
   s1+=6.0;
  }
 } else {
  if(i5<251.22314453125){
   s0+=121.0;
   s1+=2883.0;
  } else {
   s0+=545.0;
  }
 }
}
if(i12<2455.4130859375){
 if(i12<2341.7109375){
  if(i5<44.79998779296875){
   s0+=12941.0;
  } else {
   s1+=25.0;
  }
 } else {
  if(i7<94.13614654541016){
   s0+=147.0;
  } else {
   s1+=32.0;
  }
 }
} else {
 if(i5<251.7432861328125){
  if(i8<478.44598388671875){
   s0+=92.0;
   s1+=3414.0;
  } else {
   s0+=112.0;
  }
 } else {
  s0+=562.0;
 }
}
if(i5<36.41448974609375){
 s0+=13170.0;
} else {
 if(i6<6.0267333984375){
  if(i15<2397.6328125){
   s0+=66.0;
   s1+=3503.0;
  } else {
   s0+=41.0;
   s1+=14.0;
  }
 } else {
  s0+=531.0;
 }
}
if(i6<0.863525390625){
 s0+=13160.0;
} else {
 if(i12<4908.125){
  if(i8<418.9695739746094){
   s0+=12.0;
   s1+=3418.0;
  } else {
   s0+=164.0;
   s1+=5.0;
  }
 } else {
  if(i5<209.2733154296875){
   s0+=1.0;
   s1+=35.0;
  } else {
   s0+=514.0;
   s1+=16.0;
  }
 }
}
if(i9<3.76715087890625){
 if(i16<2679.5){
  if(i0<2634.8798828125){
   s0+=13033.0;
   s1+=35.0;
  } else {
   s1+=10.0;
  }
 } else {
  if(i11<2048.343017578125){
   s0+=85.0;
  } else {
   s0+=3.0;
   s1+=233.0;
  }
 }
} else {
 if(i5<251.7432861328125){
  if(i11<2074.042724609375){
   s0+=73.0;
  } else {
   s0+=96.0;
   s1+=3206.0;
  }
 } else {
  s0+=551.0;
 }
}
if(i16<2560.0){
 if(i5<42.95904541015625){
  s0+=13055.0;
 } else {
  if(i16<2558.0){
   s0+=1.0;
   s1+=99.0;
  } else {
   s0+=1.0;
  }
 }
} else {
 if(i12<4851.22314453125){
  if(i6<0.832275390625){
   s0+=164.0;
  } else {
   s0+=118.0;
   s1+=3266.0;
  }
 } else {
  if(i11<3924.7763671875){
   s0+=34.0;
   s1+=72.0;
  } else {
   s0+=515.0;
  }
 }
}
if(i4<2002.251953125){
 if(i5<39.32977294921875){
  s0+=12891.0;
 } else {
  s1+=101.0;
 }
} else {
 if(i4<3436.375){
  if(i5<3.5123291015625){
   s0+=339.0;
  } else {
   s0+=166.0;
   s1+=3335.0;
  }
 } else {
  s0+=493.0;
 }
}
if(i6<0.8548583984375){
 s0+=13238.0;
} else {
 if(i0<5171.57421875){
  if(i6<6.0687255859375){
   s0+=65.0;
   s1+=3433.0;
  } else {
   s0+=113.0;
  }
 } else {
  if(i4<3101.67431640625){
   s0+=3.0;
   s1+=9.0;
  } else {
   s0+=460.0;
   s1+=4.0;
  }
 }
}
if(i16<2628.0){
 if(i11<2215.0185546875){
  if(i16<2421.5){
   s0+=12741.0;
   s1+=20.0;
  } else {
   s0+=266.0;
   s1+=36.0;
  }
 } else {
  if(i15<2261.591552734375){
   s1+=66.0;
  } else {
   s0+=4.0;
  }
 }
} else {
 if(i6<6.0220947265625){
  if(i14<157.4234619140625){
   s0+=109.0;
   s1+=3206.0;
  } else {
   s0+=138.0;
   s1+=177.0;
  }
 } else {
  s0+=562.0;
 }
}
if(i6<0.8548583984375){
 s0+=13244.0;
} else {
 if(i6<6.0220947265625){
  if(i13<124.962158203125){
   s0+=51.0;
   s1+=3321.0;
  } else {
   s0+=51.0;
   s1+=86.0;
  }
 } else {
  s0+=572.0;
 }
}
if(i6<0.854736328125){
 s0+=13096.0;
} else {
 if(i11<3924.7763671875){
  if(i16<5199.5){
   s0+=79.0;
   s1+=3563.0;
  } else {
   s0+=40.0;
   s1+=17.0;
  }
 } else {
  s0+=530.0;
 }
}
if(i5<35.54168701171875){
 s0+=13257.0;
} else {
 if(i5<251.7432861328125){
  if(i15<2684.984619140625){
   s0+=78.0;
   s1+=3435.0;
  } else {
   s0+=21.0;
  }
 } else {
  s0+=534.0;
 }
}
if(i0<2573.718505859375){
 if(i11<2215.0185546875){
  if(i12<2491.021484375){
   s0+=13105.0;
   s1+=31.0;
  } else {
   s0+=6.0;
   s1+=10.0;
  }
 } else {
  if(i7<269.84869384765625){
   s1+=40.0;
  } else {
   s0+=3.0;
  }
 }
} else {
 if(i11<3922.1435546875){
  if(i7<282.0853271484375){
   s0+=114.0;
   s1+=3355.0;
  } else {
   s0+=121.0;
   s1+=1.0;
  }
 } else {
  s0+=539.0;
 }
}
if(i6<0.85406494140625){
 s0+=13250.0;
} else {
 if(i6<6.0203857421875){
  if(i16<5374.0){
   s0+=75.0;
   s1+=3451.0;
  } else {
   s0+=22.0;
   s1+=1.0;
  }
 } else {
  s0+=526.0;
 }
}
if(i11<2191.5048828125){
 if(i10<57.689697265625){
  if(i6<0.9913330078125){
   s0+=13140.0;
  } else {
   s1+=35.0;
  }
 } else {
  if(i5<38.87164306640625){
   s0+=30.0;
  } else {
   s1+=54.0;
  }
 }
} else {
 if(i11<3924.2412109375){
  if(i5<3.7464599609375){
   s0+=18.0;
  } else {
   s0+=90.0;
   s1+=3428.0;
  }
 } else {
  s0+=530.0;
 }
}
if(i11<2191.630859375){
 if(i6<0.952392578125){
  s0+=13014.0;
 } else {
  s1+=96.0;
 }
} else {
 if(i6<6.0220947265625){
  if(i8<469.1168212890625){
   s0+=25.0;
   s1+=3532.0;
  } else {
   s0+=122.0;
  }
 } else {
  s0+=536.0;
 }
}
if(i6<0.863525390625){
 s0+=13183.0;
} else {
 if(i5<251.7432861328125){
  if(i7<280.7758483886719){
   s0+=1.0;
   s1+=3488.0;
  } else {
   s0+=105.0;
  }
 } else {
  s0+=548.0;
 }
}
if(i5<36.00360107421875){
 s0+=13198.0;
} else {
 if(i11<3922.181640625){
  if(i16<5159.0){
   s0+=69.0;
   s1+=3476.0;
  } else {
   s0+=41.0;
   s1+=26.0;
  }
 } else {
  s0+=515.0;
 }
}
if(i16<2559.5){
 if(i7<115.67179870605469){
  if(i0<2624.32763671875){
   s0+=12355.0;
  } else {
   s1+=2.0;
  }
 } else {
  if(i8<381.880615234375){
   s1+=121.0;
  } else {
   s0+=673.0;
  }
 }
} else {
 if(i6<6.0177001953125){
  if(i6<0.832275390625){
   s0+=165.0;
  } else {
   s0+=103.0;
   s1+=3385.0;
  }
 } else {
  s0+=521.0;
 }
}
if(i5<35.520751953125){
 s0+=13157.0;
} else {
 if(i11<3922.1435546875){
  if(i16<5289.0){
   s0+=73.0;
   s1+=3479.0;
  } else {
   s0+=39.0;
   s1+=10.0;
  }
 } else {
  if(i5<232.5225830078125){
   s1+=1.0;
  } else {
   s0+=566.0;
  }
 }
}
if(i10<55.18218994140625){
 if(i6<0.8690185546875){
  s0+=13098.0;
 } else {
  if(i0<2676.2431640625){
   s0+=5.0;
   s1+=42.0;
  } else {
   s0+=2.0;
   s1+=240.0;
  }
 }
} else {
 if(i4<3436.63671875){
  if(i5<38.38922119140625){
   s0+=99.0;
  } else {
   s0+=167.0;
   s1+=3174.0;
  }
 } else {
  s0+=498.0;
 }
}
if(i5<36.02569580078125){
 s0+=13244.0;
} else {
 if(i6<6.0172119140625){
  if(i0<5212.67431640625){
   s0+=69.0;
   s1+=3416.0;
  } else {
   s0+=47.0;
   s1+=10.0;
  }
 } else {
  s0+=539.0;
 }
}
if(i16<2642.0){
 if(i5<36.33673095703125){
  s0+=13146.0;
 } else {
  if(i15<2174.205810546875){
   s1+=132.0;
  } else {
   s0+=4.0;
  }
 }
} else {
 if(i0<5154.833984375){
  if(i8<434.30889892578125){
   s0+=117.0;
   s1+=3328.0;
  } else {
   s0+=154.0;
  }
 } else {
  if(i15<190.87246704101562){
   s0+=7.0;
   s1+=9.0;
  } else {
   s0+=420.0;
   s1+=8.0;
  }
 }
}
if(i4<2002.916748046875){
 if(i5<39.32977294921875){
  s0+=12849.0;
 } else {
  s1+=66.0;
 }
} else {
 if(i5<251.7060546875){
  if(i6<0.1025390625){
   s0+=357.0;
  } else {
   s0+=102.0;
   s1+=3413.0;
  }
 } else {
  s0+=538.0;
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
