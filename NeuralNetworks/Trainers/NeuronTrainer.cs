using System;
using System.Linq;
using NeuralNetworks.NeuronWorks;

namespace NeuralNetworks.Trainers
{
    public class NeuronTrainer
    {
        private readonly Neuron _neuron;
        private readonly Func<double, double> _activator = NeuronFunctions.Activators.Linear;
        private readonly Func<double, double, double, double> _errorPrime = NeuronFunctions.Derivatives.Linear;
        private readonly double[] _examples;
        private readonly double[][] _inputs;

        public NeuronTrainer(Neuron neuron, (double[] results, double[][] inputs) data)
        {
            _neuron = neuron;
            _examples = data.results;
            _inputs = data.inputs;
        }

        private void TrainNeuron(double eta, out double error)
        {
            error = 0d;
            for (int i = 0; i < _examples.Length; i++)
            {
                var xK = _inputs[i];
                var yK = _examples[i];

                _neuron.SetInput(xK);
                var oK = _neuron.Activate();
                
                var eK = 0.5d * Math.Pow(yK-oK, 2);

                error += eK;

                CorrectWeights(eta, yK, oK, xK);
            }
        }

        private void CorrectWeights(double eta, double yK, double oK, double[] xK)
        {
            for (int i = 0; i < _neuron._weights.Length; i++)
            {
                var value = _neuron._weights[i] - eta * _errorPrime(yK, oK, xK[i]);
                _neuron.SetWeight(i, value);
            }
        }

        public void Train(int epochCount, double eta)
        {
            for (int i = 0; i < epochCount; i++)
            {
                TrainNeuron(eta, out var error);
                Console.WriteLine($"Epoch: {i}.\tDeviation: {error}");
            }
        }
    }
}