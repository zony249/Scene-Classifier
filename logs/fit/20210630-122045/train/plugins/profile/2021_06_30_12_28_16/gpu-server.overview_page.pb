?	?Xm??xz@?Xm??xz@!?Xm??xz@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?Xm??xz@U???f@1??4̭n@A[?}s??I??1?q??rEagerKernelExecute 0*	Zd;?g??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?2ı.0@@!"
ħ?X@)?2ı.0@@1"
ħ?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?gB?Ē??!?0V????)?gB?Ē??1?0V????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismjhwH1??!f??jZ???)???????1B-/N??:Preprocessing2F
Iterator::Model<? ???!????˻?)???8m?1?lỘ???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap???dp0@@!?z??X@)??-?l`?1mS?Y%Vy?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 41.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??H?E@Q}?<??L@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	U???f@U???f@!U???f@      ??!       "	??4̭n@??4̭n@!??4̭n@*      ??!       2	[?}s??[?}s??![?}s??:	??1?q????1?q??!??1?q??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??H?E@y}?<??L@?"k
=gradient_tape/model_13/conv2d_356/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterc??k㲲?!c??k㲲?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdamO?}?????!wj?b o??"k
=gradient_tape/model_13/conv2d_355/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?i?"
?!y??uѯ??08"\
;gradient_tape/model_13/max_pooling2d_52/MaxPool/MaxPoolGradMaxPoolGrad????ϙ?!pXL????"8
model_13/conv2d_374/Conv2DConv2D???a????!?;x????0"8
model_13/conv2d_377/Conv2DConv2DIׄ=,:??!???????0"i
=gradient_tape/model_13/conv2d_368/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?,9????!??????0"8
model_13/conv2d_368/Conv2DConv2D??Q?V(??!F?x?]??0"i
=gradient_tape/model_13/conv2d_362/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?4*@????!X?p??
??0"g
<gradient_tape/model_13/conv2d_374/Conv2D/Conv2DBackpropInputConv2DBackpropInput?r?K?h??!??-+a??0I9N??;A@Qc??bP@Y?c5?25??a8?????X@qIVM?Sy6@yP??MP?"?	
both?Your program is POTENTIALLY input-bound because 41.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?22.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 