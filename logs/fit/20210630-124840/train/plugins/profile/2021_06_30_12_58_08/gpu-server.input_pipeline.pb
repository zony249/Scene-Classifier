	k?ѯz@k?ѯz@!k?ѯz@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCk?ѯz@!??^??e@1l_@/?3n@A?????N??I?q?P???rEagerKernelExecute 0*	??x?N??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?֤?!3@!I????X@)?֤?!3@1I????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??,`??!v n?V???)??,`??1v n?V???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismȲ`⏢??!a;????)?߽?Ƅ??1?$????:Preprocessing2F
Iterator::Model2??8*7??!&?Y??t??)?L?x$^n?1#???xΓ?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?۞ ?!3@!????X@)??@??c?1??9??Ή?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 41.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI^~S??D@Q?????M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	!??^??e@!??^??e@!!??^??e@      ??!       "	l_@/?3n@l_@/?3n@!l_@/?3n@*      ??!       2	?????N???????N??!?????N??:	?q?P????q?P???!?q?P???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q^~S??D@y?????M@