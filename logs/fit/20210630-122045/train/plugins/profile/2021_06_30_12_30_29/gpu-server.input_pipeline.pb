	???~??x@???~??x@!???~??x@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC???~??x@??O?Tc@1z??-n@A?c?~???Ii????@rEagerKernelExecute 0*	??~jtk?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?;????@@!?(?R}?X@)?;????@@1?(?R}?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch????V%??!yT9?b~??)????V%??1yT9?b~??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism-&?(???!Fc?u??)?<Y????1rI???:Preprocessing2F
Iterator::Model`!sePm??!??˸?l??)??f??o?1ڬQ?m??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapA?;?@@!????X@)ޭ,?Yfa?1?%??y?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?Kf{?C@Q?NN@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??O?Tc@??O?Tc@!??O?Tc@      ??!       "	z??-n@z??-n@!z??-n@*      ??!       2	?c?~????c?~???!?c?~???:	i????@i????@!i????@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?Kf{?C@y?NN@