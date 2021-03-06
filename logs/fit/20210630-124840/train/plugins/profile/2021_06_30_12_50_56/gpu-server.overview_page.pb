?	?????>y@?????>y@!?????>y@	"(??7}??"(??7}??!"(??7}??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?????>y@*?Z^??c@1???d)@n@AB]¡???I????(@??Y?????ļ?rEagerKernelExecute 0*	@`??:??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator{?v??2@!VB??X@){?v??2@1VB??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchO???ʒ?!??[ð??)O???ʒ?1??[ð??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism`??V?I??!?xӨ????)#??u???15߼??l??:Preprocessing2F
Iterator::Modelqt???!??Ka?B??)? :vp?1tm??š??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??????2@!-Zψ^?X@)4??`[?1%??B΁?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9!(??7}??I+Q??D@Q?þu??M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	*?Z^??c@*?Z^??c@!*?Z^??c@      ??!       "	???d)@n@???d)@n@!???d)@n@*      ??!       2	B]¡???B]¡???!B]¡???:	????(@??????(@??!????(@??B      ??!       J	?????ļ??????ļ?!?????ļ?R      ??!       Z	?????ļ??????ļ?!?????ļ?b      ??!       JGPUY!(??7}??b q+Q??D@y?þu??M@?"k
=gradient_tape/model_32/conv2d_869/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??h?ݲ?!??h?ݲ?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam??AF?T??!E?$
???"k
=gradient_tape/model_32/conv2d_868/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter/R?Rj??!?L?\????08"]
<gradient_tape/model_32/max_pooling2d_128/MaxPool/MaxPoolGradMaxPoolGradj?V?9??!
??'0???"8
model_32/conv2d_887/Conv2DConv2D'?崪??!?nMĆ???0"8
model_32/conv2d_890/Conv2DConv2D????J??!"B??????0"8
model_32/conv2d_881/Conv2DConv2D???E??!V#(???0"i
=gradient_tape/model_32/conv2d_875/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??w'??!578^??0"i
=gradient_tape/model_32/conv2d_881/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??????!?ˍ????0"k
=gradient_tape/model_32/conv2d_890/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???????!.???V??08I???#8A@Q?q"?cP@Y?c5?25??a8?????X@q_re
?G@y??????P?"?
both?Your program is POTENTIALLY input-bound because 39.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Ampere)(: B 