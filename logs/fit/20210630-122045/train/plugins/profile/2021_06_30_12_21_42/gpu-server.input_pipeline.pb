	?N>=?>y@?N>=?>y@!?N>=?>y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?N>=?>y@
p??c@1?)?:{n@A%????I1??PN???rEagerKernelExecute 0*	L7?A??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??o|A@!f?h??X@)??o|A@1f?h??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?	??$>??!??ID????)?	??$>??1??ID????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??3g}ʡ?!?&?.h??)=?U?????1?b?????:Preprocessing2F
Iterator::Model???vi??!??o?̸??)?=?N??i?1?????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapk???|A@!????X@)Q?B?y?_?1?{?v?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI+D?z}?C@Qջ}??/N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	
p??c@
p??c@!
p??c@      ??!       "	?)?:{n@?)?:{n@!?)?:{n@*      ??!       2	%????%????!%????:	1??PN???1??PN???!1??PN???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q+D?z}?C@yջ}??/N@