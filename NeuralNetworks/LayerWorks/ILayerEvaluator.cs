using System.Collections.Generic;
using NeuralNetworks.NeuronWorks;

namespace NeuralNetworks.LayerWorks
{
    public interface ILayerEvaluator
    {
        public double[] Evaluate(Neuron[] neurons);
    }
}