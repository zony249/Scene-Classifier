	Ҩ?Io@Ҩ?Io@!Ҩ?Io@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCҨ?Io@>??WX?@1?w?Rgn@A??8G??I?z0)>>??rEagerKernelExecute 0*	Yd;???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorQJVգ?@!??ʫ?X@)QJVգ?@1??ʫ?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??	.VԐ?!??C????)??	.VԐ?1??C????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism'?y?3M??!?k,?E???)???!???11Ɍ$????:Preprocessing2F
Iterator::Model?,D????! ?f??);?/K;5g?1????Q??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?!S>??@!@ [???X@)#??fF?Z?1?p?t?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI ?~?<	??Q??ۃX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	>??WX?@>??WX?@!>??WX?@      ??!       "	?w?Rgn@?w?Rgn@!?w?Rgn@*      ??!       2	??8G????8G??!??8G??:	?z0)>>???z0)>>??!?z0)>>??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q ?~?<	??y??ۃX@