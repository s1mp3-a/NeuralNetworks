using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetworks.NeuronWorks;

namespace NeuralNetworks.NetWorks.FullyConnectedNetwork
{
    public class FullyConnectedNetwork
    {
        private int _size;
        private List<Neuron> _neurons = new();
        internal FunctionTypeTuple _functionTuple;

        public FullyConnectedNetwork(int size, FunctionTypeTuple functionTypeTuple = null)
        {
            _size = size;
            _functionTuple = functionTypeTuple;
            
            for (int i = 0; i < size; i++)
            {
                _neurons.Add(new(size, functionTypeTuple?.Activator));
            }
        }

        private void UpdateNeuronInput(double[] input)
        {
            foreach (var n in _neurons)
            {
                n.SetInput(input);
            }
        }

        public double[] ComputeOnce(double[] input)
        {
            var output = _neurons.Select(n => n.Activate()).ToArray();
            UpdateNeuronInput(output);
            return output;
        }
    }
}