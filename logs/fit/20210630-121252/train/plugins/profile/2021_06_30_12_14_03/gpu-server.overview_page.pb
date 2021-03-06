?	??3Yz@??3Yz@!??3Yz@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??3Yz@???d@1q?i]?o@A???Д???IDԷ̩??rEagerKernelExecute 0*	?5^????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator???J#z>@!L??f??X@)???J#z>@1L??f??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?<???!???I???)?<???1???I???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismo?KS8??!k.?W????)?5#??E??1Uf??g???:Preprocessing2F
Iterator::Model??L0?k??!????????)?2nj??l?1?i6?????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??ߠ?z>@!???E?X@)	3m??Jc?1eM]?R??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI(Sn??C@Q????8N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???d@???d@!???d@      ??!       "	q?i]?o@q?i]?o@!q?i]?o@*      ??!       2	???Д??????Д???!???Д???:	DԷ̩??DԷ̩??!DԷ̩??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q(Sn??C@y????8N@?"k
=gradient_tape/model_11/conv2d_302/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????<ǳ?!????<ǳ?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam????U??!????(r??"k
=gradient_tape/model_11/conv2d_301/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter~?pqj??!??3?b???08"8
model_11/conv2d_314/Conv2DConv2DNH{@I???!??C?+f??0"8
model_11/conv2d_320/Conv2DConv2D??t????!???"*??0"\
;gradient_tape/model_11/max_pooling2d_44/MaxPool/MaxPoolGradMaxPoolGradt9Ҫ7??!6L|	???"8
model_11/conv2d_323/Conv2DConv2D<??W??!?Do????0"k
=gradient_tape/model_11/conv2d_320/Conv2D/Conv2DBackpropFilterConv2DBackpropFiltery???????!??b?ra??08"k
=gradient_tape/model_11/conv2d_323/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter9???.??!?QК??08"i
<gradient_tape/model_11/conv2d_302/Conv2D/Conv2DBackpropInputConv2DBackpropInput??>jj??!?v?T??08IB?e2?@@Q_?:???P@Y?c5?25??a8?????X@q?d??ei@y???Q?P?"?
both?Your program is POTENTIALLY input-bound because 39.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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