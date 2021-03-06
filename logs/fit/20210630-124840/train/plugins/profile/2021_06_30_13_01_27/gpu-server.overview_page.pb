?	?k??\z@?k??\z@!?k??\z@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?k??\z@??̦f@1?-?en@A?I?U؜?I????U??rEagerKernelExecute 0*	B`?к??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator???`?)<@!?????X@)???`?)<@1?????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?)X?l:??!??$|)??)?)X?l:??1??$|)??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???N??!E??Mˢ??)B]¡??1???w???:Preprocessing2F
Iterator::Model?r/0+??!DJZ?^H??)Hk:!tp?1????,??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?uoEb*<@!m?\?m?X@)臭???c?1\??
c??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 41.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIM?F??,E@Q?t?U?L@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??̦f@??̦f@!??̦f@      ??!       "	?-?en@?-?en@!?-?en@*      ??!       2	?I?U؜??I?U؜?!?I?U؜?:	????U??????U??!????U??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qM?F??,E@y?t?U?L@?"l
>gradient_tape/model_51/conv2d_1382/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter_???dܲ?!_???dܲ?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam*?R?1??!*?E?????"l
>gradient_tape/model_51/conv2d_1381/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter*?_s:??!????}???08"]
<gradient_tape/model_51/max_pooling2d_204/MaxPool/MaxPoolGradMaxPoolGradA???=???!"?N%??"9
model_51/conv2d_1403/Conv2DConv2DSV'?p??!???(???0"9
model_51/conv2d_1400/Conv2DConv2D*_?@j??!ѩh?p???0"9
model_51/conv2d_1394/Conv2DConv2D??+?_[??!q?ܬ??0"j
>gradient_tape/model_51/conv2d_1388/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter? ɋ??!?B17.o??0"j
>gradient_tape/model_51/conv2d_1394/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterQR2????!lƫ????0"l
>gradient_tape/model_51/conv2d_1400/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???[????!7Fj??a??08I?O?.?CA@QX??=^P@Y?c5?25??a8?????X@qvƁ?{0@y??9??sP?"?	
both?Your program is POTENTIALLY input-bound because 41.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?16.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 