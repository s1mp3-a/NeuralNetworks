using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks
{
    public class SequentialLayerEvaluator : ILayerEvaluator
    {
        public double[] Evaluate(List<Neuron> neurons)
        {
            return neurons.Select(neuron => neuron.Activate()).ToArray();
        }
    }
}