using System.Collections.Generic;
using System.Linq;
using NeuralNetworks.NeuronWorks;

namespace NeuralNetworks.LayerWorks
{
    public class SequentialLayerEvaluator : ILayerEvaluator
    {
        public double[] Evaluate(List<Neuron> neurons)
        {
            return neurons.Select(neuron => neuron.Activate()).ToArray();
        }
    }
}