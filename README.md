# bachelorTTHRESH
Compression of 3d Volume Data

Code for my Bachelor Thesis with the topic of volume data compression.
An example testset "stagbeetle277x277x164.dat" is provided for testing.
Implemented are three volume compression techniques: TTHRESH, core Truncation and a hybrid between TTHRSH and core Truncation.
Call with two command line arguments: 1) name of data file (e.g. targetError) 
                                      2) target Error for compression (e.g. 0.0003).
After parsing of the original data, one of the three compression techniques can be selected.
