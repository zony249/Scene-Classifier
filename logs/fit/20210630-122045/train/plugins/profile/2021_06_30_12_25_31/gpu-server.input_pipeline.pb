	??9?x@??9?x@!??9?x@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??9?x@??;??cc@1?1Y?Gn@A1?䠄??I????40??rEagerKernelExecute 0*	V-?9?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator5??a0?@@!??ڞY?X@)5??a0?@@1??ڞY?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?I??{d??!8????-??)?I??{d??18????-??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelisma?HZ֝?!Bx?Yr??)??ZӼ???1ɻ)H;n??:Preprocessing2F
Iterator::Modelg?????!W??????)iSu?l?j?1D?[???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?Ov?@@!ى8???X@)G???R{a?1?&?h?Mz?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI1"h?!?C@Q?ݗU?[N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??;??cc@??;??cc@!??;??cc@      ??!       "	?1Y?Gn@?1Y?Gn@!?1Y?Gn@*      ??!       2	1?䠄??1?䠄??!1?䠄??:	????40??????40??!????40??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q1"h?!?C@y?ݗU?[N@