	#?	???y@#?	???y@!#?	???y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC#?	???y@~?ƃ-d@1@?]o@A?????"??I@??T?@rEagerKernelExecute 0*	;?O????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?n?1?<@!D???-?X@)?n?1?<@1D???-?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??m??!0???;???)??m??10???;???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismm???"??!?6k^c??)[|
????1|????Ϧ?:Preprocessing2F
Iterator::Model?	??bՠ?!?k?z??)?J???>l?1^y?ᢻ??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap*?D/??<@!?eK??X@)\?J?`?16ƾ???|?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?$????C@Qo?]9WN@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	~?ƃ-d@~?ƃ-d@!~?ƃ-d@      ??!       "	@?]o@@?]o@!@?]o@*      ??!       2	?????"???????"??!?????"??:	@??T?@@??T?@!@??T?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?$????C@yo?]9WN@