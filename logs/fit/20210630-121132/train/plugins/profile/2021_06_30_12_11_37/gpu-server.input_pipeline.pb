	???՝?y@???՝?y@!???՝?y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC???՝?y@???`??c@1?DJ?yXn@A?3??O??I????Xg @rEagerKernelExecute 0*	??"?Y??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??ؙB;3@!???X@)??ؙB;3@1???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?>#K??!JѬ????)?>#K??1JѬ????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?`???p??!?0????)p??-??1?J?^T??:Preprocessing2F
Iterator::Model???3ڪ??!F???????)????y?1u?????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap???`?;3@!;1????X@)??{?qY?1c?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIڤ????C@Q&[N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???`??c@???`??c@!???`??c@      ??!       "	?DJ?yXn@?DJ?yXn@!?DJ?yXn@*      ??!       2	?3??O???3??O??!?3??O??:	????Xg @????Xg @!????Xg @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qڤ????C@y&[N@