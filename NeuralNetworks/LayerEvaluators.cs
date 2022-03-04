using System;

namespace NeuralNetworks
{
    internal static class LayerEvaluators
    {
        private static readonly Tuple<SequentialLayerEvaluator, ParallelLayerEvaluator> _singleton =
            new(new SequentialLayerEvaluator(), new ParallelLayerEvaluator());

        public static ILayerEvaluator Sequential => _singleton.Item1;
        public static ILayerEvaluator Parallel => _singleton.Item2;
    }
}