using System;

namespace NeuralNetworks.NeuronWorks
{
    public class Neuron
    {
        private readonly int _synapseCount;
        private double _bias;
        internal double[] _weights;
        private double[] _input;
        private readonly Func<double, double> _activate;

        public Neuron(int synapseCount, Func<double, double> activator = null)
        {
            _activate = activator ?? NeuronFunctions.Activators.Sigmoid;
            _synapseCount = synapseCount;
            _input = new double[_synapseCount];
            _weights = new double[_synapseCount];
            RandomizeWeightsAndBias();
        }

        private void RandomizeWeightsAndBias()
        {
            var rnd = new Random();
            
            for (int i = 0; i < _synapseCount; i++)
            {
                _weights[i] = 1 + rnd.NextDouble();
            }
            
            _bias = 0;
        }

        private double SynapseSum()
        {
            var sum = 0d;
            
            for (int i = 0; i < _synapseCount; i++)
            {
                sum += _input[i] * _weights[i];
            }
            
            sum += _bias;
            
            return sum;
        }

        internal void SetWeight(int index, double value)
        {
            _weights[index] = value;
        }

        internal void SetInput(double[] input)
        {
            _input = input;
        }

        public double Activate()
        {
            var res = SynapseSum();
            return _activate(res);
        }
    }
}