using NeuralNetworks.NeuronWorks;

namespace NeuralNetworks.LayerWorks;

public class RbLayer
{
    private readonly RbNeuron[] _neurons;
    public int NeuronCount => _neurons.Length;
    public double[] Output { get; private set; }

    public RbLayer(int neuronCount, double[][] coefficients, double s)
    {
        _neurons = new RbNeuron[neuronCount];
        for (int i = 0; i < neuronCount; i++)
        {
            _neurons[i] = new RbNeuron(coefficients[i], s);
        }
    }

    public double[] EvaluateLayer(double[] inputs)
    {
        double[] res = new double[NeuronCount];
        
        for (int i = 0; i < NeuronCount; i++)
        {
            res[i] = _neurons[i].Activate(inputs);
        }

        Output = res;
        
        return res;
    }
}