using NeuralNetworks.LayerWorks;
using NeuralNetworks.NeuronWorks;

namespace NeuralNetworks.NetWorks.RbNetwork;

public class RbNetwork
{
    internal readonly RbLayer rbLayer;
    internal readonly Layer neuronLayer;

    private readonly double[][] _targets;
    private readonly double[][] _inputs;
    

    public RbNetwork((double[][] inputs, double[][] results) data, double s)
    {
        _inputs = data.inputs;
        _targets = data.results;
        rbLayer = new RbLayer(_inputs.GetLength(0), _inputs, s);
        neuronLayer = new Layer(_targets[0].Length,rbLayer.NeuronCount, NeuronFunctions.Functions.Linear);
    }

    public double[] GetResult(double[] input)
    {
        var output = rbLayer.EvaluateLayer(input);
        neuronLayer.SetInputs(output);
        return neuronLayer.EvaluateLayer();
    }

    public void Train(int epochCount, double learningRate)
    {
        for (int i = 0; i < epochCount; i++)
        {
            TrainNetwork(learningRate, out double error);
            Console.WriteLine($"Progress: {i}/{epochCount}.\tError:{error}");
        }

        for (int i = 0; i < _inputs.GetLength(0); i++)
        {
            var netRes = this.GetResult(_inputs[i]);
            string input = $"[{string.Join(';', _inputs[i])}]";
            string result = $"[{string.Join(';', netRes)}]";
            string expected = $"[{string.Join(';', _targets[i])}]";
            Console.WriteLine($"For input: {input}    got: {result}    expected: {expected}");
        }
    }

    private void TrainNetwork(double learningRate, out double error)
    {
        error = 0d;
        for (int targetIdx = 0; targetIdx < _targets.Length; targetIdx++)
        {
            this.GetResult(_inputs[targetIdx]);
            for (int neuronIdx = 0; neuronIdx < neuronLayer.NeuronCount; neuronIdx++)
            {
                error += TrainNeuron(learningRate, targetIdx, neuronIdx);
            }
        }
    }

    private double TrainNeuron(double learningRate, int targetIdx, int neuronIdx)
    {
        var xK = rbLayer.Output;
        var yk = _targets[targetIdx][neuronIdx];

        var neuron = neuronLayer.Neurons[neuronIdx];
        neuron.SetInput(xK);
        var oK = neuron.Activate();

        var ek = 0.5d * Math.Pow(yk-oK, 2);

        for (int i = 0; i < neuron._weights.Length; i++)
        {
            neuron._weights[i] = neuron._weights[i] - learningRate * (yk - oK) * (-xK[i]);
        }

        return ek;
    }
}