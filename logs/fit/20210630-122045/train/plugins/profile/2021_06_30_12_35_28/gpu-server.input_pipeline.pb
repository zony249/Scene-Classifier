	M,??$z@M,??$z@!M,??$z@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCM,??$z@????d@1???sn?o@An?|?b???I?
DO? @rEagerKernelExecute 0*	~?5^2#?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator2?F??;@!Nߧ???X@)2?F??;@1Nߧ???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???W?ё?!>1?????)???W?ё?1>1?????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismⱟ?R$??!n?a????)???ۂ???1`ڪ?????:Preprocessing2F
Iterator::Modell?p?握?!??Ej???)?w???o?1?!
mU???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap4J??%?;@!G?n??X@)2: 	?vb?1?(?ۘ???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIhT???C@Q????!N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????d@????d@!????d@      ??!       "	???sn?o@???sn?o@!???sn?o@*      ??!       2	n?|?b???n?|?b???!n?|?b???:	?
DO? @?
DO? @!?
DO? @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qhT???C@y????!N@