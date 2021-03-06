?	??A ?y@??A ?y@!??A ?y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??A ?y@ݖ???d@1@3??nn@A6\䞮???Ilw?}? @rEagerKernelExecute 0*	>
ף???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator???WG3@![]?>?X@)???WG3@1[]?>?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??Z
H???!????????)??Z
H???1????????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??**???!"??Ϗy??)??#??1;?rL??:Preprocessing2F
Iterator::Model'???C??!.Py?<??)???jq?1cu???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapu ???G3@!XCv???X@)`x%?s}_?1???%#b??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 40.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIskQD@Q??????M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ݖ???d@ݖ???d@!ݖ???d@      ??!       "	@3??nn@@3??nn@!@3??nn@*      ??!       2	6\䞮???6\䞮???!6\䞮???:	lw?}? @lw?}? @!lw?}? @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qskQD@y??????M@?"l
>gradient_tape/model_50/conv2d_1355/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterp?p???!p?p???08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam?(y1???!?N?T???"l
>gradient_tape/model_50/conv2d_1354/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?FZ?w3??!%pJ???08"]
<gradient_tape/model_50/max_pooling2d_200/MaxPool/MaxPoolGradMaxPoolGradƈlF??!>!M????"9
model_50/conv2d_1373/Conv2DConv2D???ￛ??!??KS???0"9
model_50/conv2d_1376/Conv2DConv2DF?ֺAv??!ܵt????0"9
model_50/conv2d_1367/Conv2DConv2Dc@A????!????P???0"j
>gradient_tape/model_50/conv2d_1361/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??p?2??!????g??0"j
>gradient_tape/model_50/conv2d_1367/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterJ????!i??[R??0"l
>gradient_tape/model_50/conv2d_1376/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?F??????!?|???^??08I???+!4A@Q?j?eP@Y?c5?25??a8?????X@qN`?6??9@y,?Е?nP?"?	
both?Your program is POTENTIALLY input-bound because 40.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?26.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 