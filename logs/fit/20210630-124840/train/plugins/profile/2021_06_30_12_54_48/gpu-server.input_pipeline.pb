	???~/y@???~/y@!???~/y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC???~/y@?]??c@1F?x?/n@AS??.???IZ????#@rEagerKernelExecute 0*	D?l????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorB#ظ?I=@!WL&ki?X@)B#ظ?I=@1WL&ki?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?<???!??w? ??)?<???1??w? ??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismqZ??? ??!MX??1ո?)?;l"3??1???׉??:Preprocessing2F
Iterator::Model??8~??!??Q?N??)????n?1(T???P??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap:??*?J=@!??Z,??X@)<?.9?d?1??&؁?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noII????D@Q?[mj?M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?]??c@?]??c@!?]??c@      ??!       "	F?x?/n@F?x?/n@!F?x?/n@*      ??!       2	S??.???S??.???!S??.???:	Z????#@Z????#@!Z????#@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qI????D@y?[mj?M@