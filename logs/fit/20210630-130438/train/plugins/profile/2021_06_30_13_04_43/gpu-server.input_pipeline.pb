	??c???x@??c???x@!??c???x@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??c???x@??O?c@1pu ??jn@A???n?ڟ?IU???????rEagerKernelExecute 0*	U㥛l??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?G??=@!c{???X@)?G??=@1c{???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchE7????!1iV?#???)E7????11iV?#???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?2???!?P?񚄸?)?Hh˹??1~8?>p??:Preprocessing2F
Iterator::Model????aN??![???v??)?????k?1???ݒ??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap????b?=@!<?UB"?X@)	???W_?1?6??6ez?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?!??kC@Q?4??N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??O?c@??O?c@!??O?c@      ??!       "	pu ??jn@pu ??jn@!pu ??jn@*      ??!       2	???n?ڟ????n?ڟ?!???n?ڟ?:	U???????U???????!U???????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?!??kC@y?4??N@