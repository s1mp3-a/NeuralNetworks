using System.Collections.Generic;

namespace NeuralNetworks
{
    public interface ILayerEvaluator
    {
        public double[] Evaluate(List<Neuron> neurons);
    }
}