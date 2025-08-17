

# **A Deep Learning Autoencoder Framework for End-to-End OFDM Modem Design**

## **I. State-of-the-Art in Learned Physical Layer Communications**

### **1.1 The Paradigm Shift: From Modular to End-to-End Physical Layer Design**

The architecture of conventional wireless communication systems is a testament to decades of rigorous mathematical modeling and optimization. This traditional approach is characterized by a modular, block-based structure, where the physical layer (PHY) is decomposed into a series of independent processing blocks, each responsible for a distinct function such as source coding, channel coding, modulation, channel estimation, and equalization.1 While this modular design has been instrumental in the development of highly efficient and stable systems, its fundamental premise—that the local optimization of each block leads to a globally optimal system—is not guaranteed.3 This component-wise optimization introduces artificial barriers and can result in suboptimal end-to-end performance, particularly when faced with complex channel conditions or hardware impairments that are not perfectly captured by the underlying mathematical models.1

In recent years, the confluence of powerful computational hardware and advances in deep learning has catalyzed a paradigm shift in physical layer design. This new approach, termed end-to-end learning, reimagines the communication system not as a chain of discrete blocks, but as a single, unified system that can be optimized jointly.3 By leveraging deep neural networks (DNNs), this data-driven methodology seeks to learn the functions of the transmitter and receiver directly from data, adapting to the specific characteristics of the channel and hardware.4 This holistic optimization can overcome the limitations of traditional designs by discovering novel and robust communication strategies that are not constrained by human-derived models, thereby closing the gap between theoretical assumptions and practical system imperfections.5

The conceptual framework for this end-to-end approach is the autoencoder (AE), a class of neural networks designed for unsupervised learning of efficient data representations.6 In this context, the entire communication system is modeled as an autoencoder, where the transmitter and receiver are jointly trained to minimize a single performance metric, such as the bit error rate (BER), over a distribution of channel conditions.1 This powerful abstraction allows for the joint optimization of all trainable components via gradient-based methods, promising a new frontier of AI-native communication systems that are more adaptive, robust, and performant than their handcrafted predecessors.7

### **1.2 Autoencoders as Communication Systems: Principles and Proven Advantages**

The application of the autoencoder framework to communications provides a conceptually elegant and powerful method for joint transceiver design. The transmitter is cast as the encoder, a neural network parameterized by weights $ \\theta\_T $, which learns a function $ f\_{\\theta\_T} $ to map a message or a block of information bits $ \\mathbf{b} $ into a latent representation $ \\mathbf{x} $. This latent representation is the physical signal transmitted over the channel.1 The channel itself, with all its inherent impairments such as noise, fading, and interference, acts as a stochastic, non-trainable layer that corrupts the transmitted signal $ \\mathbf{x} $ to produce the received signal $ \\mathbf{y} $. The receiver, acting as the decoder and parameterized by weights $ \\theta\_R $, is a neural network that learns a function $ g\_{\\theta\_R} $ to reconstruct an estimate of the original information, $ \\hat{\\mathbf{b}} $, from the distorted received signal $ \\mathbf{y} $.6

The entire system is trained end-to-end by minimizing a loss function, typically the binary cross-entropy between the transmitted bits $ \\mathbf{b} $ and the receiver's output probabilities $ \\hat{\\mathbf{b}} $, using stochastic gradient descent (SGD) or its variants.3 This joint optimization process allows the encoder and decoder to co-adapt, learning a communication scheme that is inherently robust to the specific channel impairments encountered during training. The key advantage of this approach is its ability to move beyond the linear, block-based assumptions of traditional systems. The neural networks can learn complex, non-linear mappings that discover novel coding and modulation schemes, effectively learning to pre-compensate for channel distortions at the transmitter and perform sophisticated joint detection at the receiver. This has been shown to yield significant BER performance gains over conventional systems, especially in the presence of challenging impairments like severe phase noise or non-linear amplifier effects, which are difficult to model and compensate for using traditional techniques.3

A critical enabler for this end-to-end training is the requirement of a differentiable channel model.1 For the gradients of the loss function, calculated at the receiver's output, to propagate back through the receiver network and subsequently to the transmitter network, every operation in the forward path must be differentiable. While a physical channel is a non-differentiable "black box," training can be performed using a simulated, mathematically-defined channel model that accurately represents the physical phenomena and is constructed from differentiable operations.13 This allows the transmitter to learn how changes in its output signal $ \\mathbf{x} $ will affect the final loss, enabling it to discover representations that are maximally resilient to channel perturbations.

### **1.3 Specialization to OFDM: Architectures and Performance in Frequency-Selective Channels**

While the autoencoder concept is general, its application to Orthogonal Frequency Division Multiplexing (OFDM) systems has proven to be a particularly fruitful area of research. OFDM is the dominant waveform in modern wireless standards due to its ability to convert a frequency-selective multipath channel into a set of parallel, flat-fading subchannels, simplifying equalization.15 Extending the end-to-end learning paradigm to OFDM leverages this powerful structure while allowing for joint optimization of the processing blocks that traditionally surround the core OFDM modulation.3

A key finding in the literature is that a hybrid approach, which combines learned components with fixed, domain-specific signal processing blocks, often outperforms a purely "black-box" learning model.3 Specifically, for an OFDM autoencoder, the Inverse Fast Fourier Transform (IFFT) at the transmitter and the Fast Fourier Transform (FFT) at the receiver are maintained as fixed, non-trainable layers. While a sufficiently deep neural network could theoretically learn an approximation of the Fourier transform, explicitly incorporating these known-optimal operations provides a powerful inductive bias. This constrains the vast search space of the neural network, focusing its learning capacity on the aspects of the problem that lack a closed-form optimal solution, such as the joint design of channel coding and modulation. This model-driven deep learning approach results in significantly faster training convergence, reduced model complexity, and improved generalization performance.3 The resulting OFDM autoencoder thus retains the principal benefits of the conventional OFDM scheme—namely, robustness against sampling synchronization errors and the enablement of simple single-tap equalization—while gaining the performance advantages of end-to-end optimization.3

Several studies have demonstrated the efficacy of this approach. OFDM autoencoders have been shown to successfully learn communication strategies over frequency-selective fading channels, outperforming state-of-the-art OFDM baselines that use conventional modulation and coding schemes.3 By jointly optimizing the transmitter and receiver, these systems can implicitly learn to handle channel distortions and even hardware impairments, discovering novel constellation geometries and symbol mappings that are tailored to the statistics of the channel.3 This demonstrates that the OFDM framework provides a robust and effective scaffold upon which powerful, learned communication systems can be built.

### **1.4 Advanced Concepts: Neural Modulation and Information Spreading on the Resource Grid**

The evolution of learned communication systems is progressing beyond simply replacing traditional processing blocks with neural network equivalents. More advanced research is exploring how to use neural networks to fundamentally redefine the transmitted waveform itself. A prime example of this is the concept of "neural modulation," where the transmitter network performs a learned, non-linear mapping that jointly accomplishes channel coding and modulation in a sophisticated, spatially-coupled manner.

A recent and compelling framework in this domain is "Deep-OFDM".15 In this architecture, the transmitter is not a simple mapper that assigns bits to individual subcarriers. Instead, it employs a Convolutional Neural Network (CNN) to actively spread the information from a block of bits across the entire two-dimensional (2D) time-frequency resource grid.15 This approach represents a significant departure from conventional OFDM, where each resource element is modulated independently. The convolutional nature of the transmitter means that each transmitted I/Q symbol on the grid is a function of multiple input bits, and each input bit influences a region of symbols on the grid. This effectively creates a learned space-frequency code, building in redundancy and diversity gain that is explicitly optimized for a corresponding neural receiver.

This co-design of a neural modulator with a neural receiver is particularly potent. The CNN-based transmitter is optimized to produce a signal structure that aligns with the inductive biases of a CNN-based receiver, which is adept at exploiting 2D correlations.16 This synergy enables the system to achieve substantial performance gains, especially in challenging, doubly-selective channels characteristic of high-mobility scenarios where both time and frequency selectivity are high.15 In such environments, the system can learn to exploit time-frequency diversity to combat fading and inter-carrier interference (ICI). Furthermore, research has shown that these systems can learn to create implicit pilot structures within the data transmission itself, enabling reliable communication even in pilot-sparse or completely pilotless regimes, thereby dramatically improving spectral efficiency.15 This line of research signifies a deeper integration of AI into the physical layer, moving from optimizing existing structures to discovering entirely new, high-performance waveforms.

## **II. System Model and Mathematical Formulation**

### **2.1 The Conventional OFDM Baseline and Drone Communication System Parameters**

To ground the design and evaluation of the proposed deep learning-based modem, it is essential to first define the physical layer parameters of the target drone communication system. These parameters are derived from standard wireless specifications and align with the user's project requirements. They will dictate the dimensions of the data structures, the timing of the system, and the scale of the neural network architecture. The key parameters are summarized in Table 1\.

The system operates over a maximum bandwidth of 20 MHz with a subcarrier spacing of 15 kHz. This corresponds to a standard 5G NR numerology. A 10 ms frame is composed of 120 OFDM symbols. The autoencoder will operate on a slot-by-slot basis, where each slot consists of 6 OFDM symbols. The total number of subcarriers will vary depending on the operational bandwidth, scaling from approximately 200 for a 3 MHz channel to 1200 for a 20 MHz channel. The FFT size and cyclic prefix length are chosen accordingly to accommodate this structure and mitigate inter-symbol interference from expected channel delay spreads.

| Parameter | Value / Specification | Derivation / Note |
| :---- | :---- | :---- |
| **RF & Frame Structure** |  |  |
| Subcarrier Spacing ($ \\Delta f $) | 15 kHz | Standard LTE/5G NR numerology. |
| Maximum Channel Bandwidth | 20 MHz | Project requirement. |
| Frame Duration | 10 ms | Project requirement. |
| Symbols per Frame | 120 | Project requirement. |
| Slot Duration | 0.5 ms | $ (120 \\text{ symbols/frame} / 20 \\text{ slots/frame})^{-1} \\times 10 \\text{ ms} $ |
| Symbols per Slot ($ N\_{sym} $) | 6 | Project requirement for AE processing block. |
| **OFDM Parameters (20 MHz BW)** |  |  |
| FFT Size ($ N\_{FFT} $) | 2048 | Standard value for 20 MHz bandwidth. |
| Number of Active Subcarriers ($ N\_{sc} $) | 1200 | Typical for 20 MHz channel. |
| Symbol Duration ($ T\_{sym} $) | $ \\approx 71.4 \\ \\mu s $ | $ T\_u \+ T\_{cp} \= 1/\\Delta f \+ T\_{cp} $ |
| Useful Symbol Time ($ T\_u $) | $ 66.67 \\ \\mu s $ | $ 1 / (15 \\text{ kHz}) $ |
| Cyclic Prefix Length ($ N\_{cp} $) | 144 samples | Normal CP for $ N\_{FFT}=2048 $. |
| **OFDM Parameters (3 MHz BW)** |  |  |
| FFT Size ($ N\_{FFT} $) | 256 | Scaled for 3 MHz bandwidth. |
| Number of Active Subcarriers ($ N\_{sc} $) | 200 | Approximate for 3 MHz channel. |
| Cyclic Prefix Length ($ N\_{cp} $) | 18 samples | Scaled Normal CP. |
| **Autoencoder Parameters** |  |  |
| Input Data Bits per Slot ($ K $) | TBD | To be determined based on desired spectral efficiency. |
| Encoder Output Grid Size | $ N\_{sc} \\times N\_{sym} $ | e.g., $ 1200 \\times 6 $ for 20 MHz. |
| Decoder Input Grid Size | $ N\_{sc} \\times N\_{sym} $ | e.g., $ 1200 \\times 6 $ for 20 MHz. |
| Decoder Output Size | $ K $ | Number of recovered bits. |

*Table 1: System and OFDM Parameters*

### **2.2 Problem Formulation: Replacing the Receiver Chain with a Differentiable Autoencoder**

The core objective is to design an autoencoder that learns an end-to-end communication scheme for a 6-symbol OFDM slot. Let the input to the system be a vector of $ K $ information bits, denoted by $ \\mathbf{b} \\in {0, 1}^K $.

The transmitter, or encoder, is a neural network represented by the function $ f\_{\\theta\_T} $. It maps the bit vector $ \\mathbf{b} $ to a complex-valued 2D tensor $ \\mathbf{X} \\in \\mathbb{C}^{N\_{sc} \\times N\_{sym}} $, which represents the I/Q values to be mapped onto the OFDM resource grid.

X=fθT​​(b)

This tensor $ \\mathbf{X} $ constitutes the latent representation of the autoencoder.  
The channel applies a stochastic transformation, $ C(\\cdot) $, to the transmitted signal. For this system, the channel effects include frequency-selective fading, Additive White Gaussian Noise (AWGN), and a residual Carrier Frequency Offset (CFO). The received resource grid after FFT at the receiver, $ \\mathbf{Y} \\in \\mathbb{C}^{N\_{sc} \\times N\_{sym}} $, can be modeled as:

Y=C(X)+N

where $ \\mathbf{N} $ is a complex AWGN tensor.  
The receiver, or decoder, is a neural network represented by the function $ g\_{\\theta\_R} $. It takes the noisy, distorted grid $ \\mathbf{Y} $ as input and produces a vector of log-likelihoods (or probabilities) for each of the original bits, $ \\hat{\\mathbf{b}} \\in ^K $.

b^=gθR​​(Y)  
The overarching goal is to find the optimal set of network parameters, $ \\theta^\* \= {\\theta\_T^, \\theta\_R^} $, that minimize the expected loss between the transmitted and reconstructed bits over the distribution of all possible channel realizations and noise. This is formulated as an optimization problem:

θ∗=argθT​,θR​min​Eb,C,N​

where $ L(\\cdot, \\cdot) $ is a suitable loss function, such as the binary cross-entropy, and the expectation is taken over all possible messages, channel states, and noise instances.

### **2.3 The End-to-End Signal Flow: From Bits to the Latent I/Q Grid and Back**

The complete signal processing chain, integrating both learned and fixed components, is crucial for a successful implementation. The flow is as follows:

1. **Input Data Generation:** A batch of random bit vectors $ \\mathbf{b} $ of size $ K $ is generated.  
2. **Encoder (Transmitter NN):** Each bit vector $ \\mathbf{b} $ is fed into the encoder network $ f\_{\\theta\_T} $, which outputs a 2D complex tensor $ \\mathbf{X} $ of size $ N\_{sc} \\times N\_{sym} $. This tensor represents the desired constellation points on the active subcarriers for the 6-symbol slot.  
3. **Power Normalization:** The output $ \\mathbf{X} $ is normalized to enforce an average power constraint, ensuring that $ \\mathbb{E}\[|\\mathbf{X}*{k,l}|^2\] \= 1 $, where $ \\mathbf{X}*{k,l} $ is the symbol at subcarrier $ k $ and time $ l $. This is a critical step for stable training and fair performance comparison.21  
4. **OFDM Modulation (Fixed Layer):** The normalized grid $ \\mathbf{X} $ is passed to a non-trainable OFDM modulation layer. This layer performs the following standard operations:  
   * Maps the $ N\_{sc} $ active subcarrier symbols to the central bins of an $ N\_{FFT} $-point vector, with nulls on DC and guard bands.  
   * Performs an $ N\_{FFT} $-point IFFT on each of the $ N\_{sym} $ columns to transform the signal to the time domain.  
   * Prepends a cyclic prefix (CP) of length $ N\_{cp} $ to each time-domain symbol.  
   * Serializes the symbols to produce a 1D time-domain complex signal vector $ \\mathbf{x}\_{time} $.  
5. **Differentiable Channel (Stochastic Layer):** The time-domain signal $ \\mathbf{x}*{time} $ passes through the simulated channel, which applies frequency-selective fading, residual CFO, and adds AWGN. This produces the received time-domain signal $ \\mathbf{y}*{time} $.  
6. **OFDM Demodulation (Fixed Layer):** The received signal $ \\mathbf{y}\_{time} $ is processed by a non-trainable OFDM demodulation layer, which reverses the modulation steps:  
   * Deserializes the signal into $ N\_{sym} $ symbols.  
   * Removes the CP from each symbol.  
   * Performs an $ N\_{FFT} $-point FFT on each symbol to return to the frequency domain.  
   * Extracts the symbols from the $ N\_{sc} $ active subcarriers to form the received grid $ \\mathbf{Y} $.  
7. **Decoder (Receiver NN):** The received grid $ \\mathbf{Y} $ is fed into the decoder network $ g\_{\\theta\_R} $, which outputs the bit probability vector $ \\hat{\\mathbf{b}} $.  
8. **Loss Calculation and Backpropagation:** The loss is calculated between $ \\mathbf{b} $ and $ \\hat{\\mathbf{b}} $, and gradients are computed and propagated back through the entire chain to update the weights $ \\theta\_T $ and $ \\theta\_R $.

### **2.4 Differentiable Channel Modeling: Mathematical Representation of AWGN, Rayleigh Fading, and Residual CFO**

The channel model is the bridge that allows gradients to flow from the receiver to the transmitter. It must be a realistic yet differentiable representation of the physical medium. The model is constructed as a sequence of differentiable layers.

Additive White Gaussian Noise (AWGN):  
The AWGN is modeled by adding a complex random variable $ n \\sim \\mathcal{CN}(0, \\sigma\_n^2) $ to each received sample. The noise variance $ \\sigma\_n^2 $ is determined by the desired Signal-to-Noise Ratio (SNR). In a simulation framework like PyTorch, this is implemented by sampling from a standard normal distribution and scaling the result.

y=x+n

This additive operation is inherently differentiable with respect to $ \\mathbf{x} $.  
Rayleigh Fading:  
For a mobile drone application, a frequency-selective fading channel is essential. A common and effective model is the Rayleigh block fading channel, where the channel is assumed to be constant for the duration of a processing block (one slot, in this case) but varies independently from one block to the next.22 The channel's effect is modeled in the frequency domain after the receiver's FFT. For each slot, a channel frequency response vector $ \\mathbf{H} \\in \\mathbb{C}^{N\_{sc}} $ is generated. Each element $ H\_k $ of this vector is a complex Gaussian random variable, $ H\_k \\sim \\mathcal{CN}(0, 1\) $, representing the channel gain on the $ k $-th subcarrier. The received grid $ \\mathbf{Y} $ is then related to the transmitted grid $ \\mathbf{X} $ by an element-wise (Hadamard) product:  
Yk,l​=Hk​⋅Xk,l​

where $ k $ is the subcarrier index and $ l $ is the symbol index. This element-wise multiplication is a differentiable operation. The use of a statistical channel model during training serves a dual purpose. Beyond simulating the physical medium, it acts as a powerful form of data augmentation. By exposing the autoencoder to a vast number of different channel realizations, it forces the network to learn a generalized encoding and decoding scheme that is robust to the underlying statistics of the fading process, rather than overfitting to a single, deterministic channel instance. The richness of this channel model distribution is therefore directly correlated with the robustness of the final learned system.  
Residual Carrier Frequency Offset (CFO):  
Despite initial synchronization, a small residual CFO, $ \\Delta f $, often remains due to oscillator inaccuracies or high Doppler shifts.2 This offset manifests as a phase rotation that increases linearly with time. In the time domain, after the IFFT at the transmitter, the signal $ x\[n\] $ is multiplied by a complex exponential:  
xcfo​\[n\]=x\[n\]⋅ej2πΔf⋅nTs​

where $ n $ is the discrete time index and $ T\_s $ is the sampling period. This multiplication is a differentiable operation. To implement this as a differentiable layer, a random $ \\Delta f $ is sampled from a predefined distribution (e.g., uniform) for each batch during training. A time vector is created, and the corresponding complex exponential is computed and multiplied with the time-domain signal. This forces the autoencoder to learn a representation that is invariant to these small phase rotations, a task that is notoriously difficult for high-order QAM in traditional systems.

## **III. Proposed Convolutional Autoencoder Architecture**

### **3.1 Architectural Rationale: Optimality of Convolutional Autoencoders for the OFDM Resource Grid**

The selection of a neural network architecture should be guided by the intrinsic structure of the data it is designed to process. The OFDM resource grid, a 2D tensor with axes of frequency and time, shares a strong structural analogy with images in the field of computer vision. Just as adjacent pixels in an image exhibit strong correlations, adjacent resource elements in an OFDM grid are correlated due to the physics of the wireless channel. Frequency-selective fading introduces correlations along the frequency axis, where nearby subcarriers experience similar channel gains. Time-varying effects, such as residual CFO or channel aging in high-mobility scenarios, introduce correlations along the time axis.16

Convolutional Neural Networks (CNNs) are exceptionally well-suited to exploit such local, spatially-correlated structures.26 Their defining feature, the convolutional layer, uses shared, learnable filters (kernels) that slide across the input data, detecting patterns and features irrespective of their absolute position. This property, known as translation equivariance, is highly desirable for processing the OFDM grid. A CNN can learn filters that detect patterns corresponding to channel distortions—such as the characteristic slope of a phase ramp caused by CFO or the correlated amplitudes of a fading channel—across any part of the time-frequency grid. In contrast, a fully connected (dense) network would treat each resource element independently, requiring an intractable number of parameters and failing to capture the essential 2D structure of the problem. The demonstrated success of CNNs in both replacing receiver blocks and, more recently, in constructing novel transmitter modulations like Deep-OFDM, solidifies their position as the optimal architectural choice for this task.15

### **3.2 Encoder (Transmitter) Design: A Deep Dive into the Convolutional Modulator Architecture**

The encoder's role is to transform a 1D vector of $ K $ information bits into a 2D complex-valued resource grid of size $ N\_{sc} \\times N\_{sym} $. This is fundamentally a process of dimensionality increase and information spreading, analogous to image generation. Therefore, the architecture will be based on transposed convolutional layers, which are adept at upsampling and learning spatial feature generation.29

The proposed architecture, detailed in Table 2, proceeds as follows:

1. **Input Embedding:** The input bit vector $ \\mathbf{b} $ of size $ K $ is first processed by a dense layer. This layer acts as an embedding, projecting the sparse binary input into a dense, continuous-valued vector in a higher-dimensional space. This allows the network to form rich initial representations of the bit patterns.  
2. **Reshaping for Convolution:** The output of the dense layer is reshaped into a 4D tensor with spatial dimensions of $ 1 \\times 1 $ and a large number of channels (e.g., 1024). This format serves as the seed for the subsequent convolutional upsampling.  
3. **Convolutional Upsampling:** A stack of Conv2DTranspose layers is used to progressively increase the spatial dimensions of the tensor until it matches the target resource grid size ($ N\_{sc} \\times N\_{sym} $). Each layer upsamples its input and applies a set of learned filters, effectively "painting" features onto the expanding grid. Batch Normalization is applied after each layer to stabilize training by normalizing the activations, and a non-linear activation function like ReLU introduces the capacity to learn complex mappings.  
4. **Output Layer:** A final Conv2D layer with a kernel size of $ 1 \\times 1 $ and two output channels is used to produce the final grid. These two channels represent the real (I) and imaginary (Q) components of the complex symbols. A linear activation function is used here to allow the output values to be unconstrained.  
5. **Power Normalization:** The final real-valued tensor is converted to a complex tensor and passed through a custom normalization layer. This layer scales the entire grid to ensure the average power per symbol is unity, which is crucial for operating within regulatory power limits and for consistent channel modeling.21

This architecture does not merely map bits to constellation points; it learns a sophisticated, non-linear modulation scheme. The overlapping receptive fields of the transposed convolutions mean that each input bit influences a region of the output grid, and each output symbol is a function of many input bits. This creates a learned, spatially-coupled code across the time-frequency grid, inherently building in redundancy and robustness against channel impairments.

| Layer \# | Layer Type | Output Shape (for 200x6 grid) | Kernel Size | Stride | Activation | Notes |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 1 | Dense | (None, 1024\) | \- | \- | ReLU | Input embedding from K bits. |
| 2 | Reshape | (None, 1, 1, 1024\) | \- | \- | \- | Prepare for convolutional layers. |
| 3 | Conv2DTranspose | (None, 25, 1, 512\) | (25, 1\) | (1, 1\) | ReLU | Upsample frequency dimension. |
| 4 | BatchNormalization | (None, 25, 1, 512\) | \- | \- | \- | Stabilize training. |
| 5 | Conv2DTranspose | (None, 50, 3, 256\) | (4, 3\) | (2, 2\) | ReLU | Upsample both dimensions. |
| 6 | BatchNormalization | (None, 50, 3, 256\) | \- | \- | \- | Stabilize training. |
| 7 | Conv2DTranspose | (None, 100, 6, 128\) | (4, 4\) | (2, 2\) | ReLU | Upsample to final size. |
| 8 | BatchNormalization | (None, 100, 6, 128\) | \- | \- | \- | Stabilize training. |
| 9 | Conv2DTranspose | (None, 200, 6, 64\) | (4, 1\) | (2, 1\) | ReLU | Final upsample in frequency. |
| 10 | BatchNormalization | (None, 200, 6, 64\) | \- | \- | \- | Stabilize training. |
| 11 | Conv2D | (None, 200, 6, 2\) | (3, 3\) | (1, 1\) | Linear | Output I and Q channels. Padding='same'. |
| 12 | Normalization | (None, 200, 6\) | \- | \- | \- | Enforce average power constraint. Output is complex. |

*Table 2: Proposed CNN Encoder Architecture (Example for 3 MHz BW)*

### **3.3 Decoder (Receiver) Design: The Convolutional Demodulator and Bit Reconstruction Architecture**

The decoder's architecture is designed to be a near-symmetrical inverse of the encoder. Its function is to process the received, distorted grid $ \\mathbf{Y} $ and recover the original $ K $ bits. This is a task of feature extraction and classification, for which standard CNN architectures are highly effective.

The proposed decoder architecture, detailed in Table 3, is as follows:

1. **Input Processing:** The complex-valued received grid $ \\mathbf{Y} $ of size $ N\_{sc} \\times N\_{sym} $ is first split into its real (I) and imaginary (Q) components, forming a real-valued tensor of size $ N\_{sc} \\times N\_{sym} \\times 2 $. This is the standard method for feeding complex signals into real-valued neural networks.3  
2. **Convolutional Feature Extraction:** A stack of Conv2D layers processes the input grid. These layers use filters to detect spatial patterns and features indicative of the transmitted data, progressively downsampling the grid's spatial dimensions while increasing the number of feature channels. Each convolutional layer learns to recognize the characteristic signatures of the transmitted symbols as they appear after being warped by the channel.  
3. **Flattening:** After the final convolutional layer, the resulting 3D feature map is flattened into a 1D vector. This vector represents a high-level, abstract summary of the information contained in the entire received slot.  
4. **Dense Classification Head:** This feature vector is passed through one or more dense layers. These layers perform the final non-linear processing, analogous to a classifier, to disentangle the features and map them to the individual bits.  
5. **Output Layer:** The final layer is a dense layer with $ K $ output neurons, one for each transmitted bit. A sigmoid activation function is applied to this layer, constraining the outputs to the range $ $. Each output value can be interpreted as the probability that the corresponding transmitted bit was a '1'.

This architecture performs joint channel estimation, equalization, and demapping in a single, holistic operation. The convolutional filters do not explicitly estimate the channel matrix $ \\mathbf{H} $. Instead, they learn to recognize the combined patterns of $ \\mathbf{H} \\cdot \\mathbf{X} $. This allows the network to learn robust decision boundaries directly from the distorted observations, bypassing the potential for error propagation inherent in a multi-stage, model-based receiver chain.30

| Layer \# | Layer Type | Output Shape (for 200x6x2 input) | Kernel Size | Stride | Activation | Notes |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 1 | Conv2D | (None, 200, 6, 64\) | (3, 3\) | (1, 1\) | LeakyReLU | Initial feature extraction. Padding='same'. |
| 2 | BatchNormalization | (None, 200, 6, 64\) | \- | \- | \- | Stabilize training. |
| 3 | Conv2D | (None, 100, 3, 128\) | (4, 4\) | (2, 2\) | LeakyReLU | Downsample and extract features. |
| 4 | BatchNormalization | (None, 100, 3, 128\) | \- | \- | \- | Stabilize training. |
| 5 | Conv2D | (None, 50, 1, 256\) | (4, 3\) | (2, 2\) | LeakyReLU | Further downsampling. |
| 6 | BatchNormalization | (None, 50, 1, 256\) | \- | \- | \- | Stabilize training. |
| 7 | Conv2D | (None, 25, 1, 512\) | (4, 1\) | (2, 1\) | LeakyReLU | Final convolutional layer. |
| 8 | Flatten | (None, 12800\) | \- | \- | \- | Convert 2D features to 1D vector. |
| 9 | Dense | (None, 4096\) | \- | \- | LeakyReLU | High-level feature processing. |
| 10 | Dropout | (None, 4096\) | \- | \- | \- | Regularization (rate=0.25). |
| 11 | Dense | (None, K) | \- | \- | Sigmoid | Output K bit probabilities. |

*Table 3: Proposed CNN Decoder Architecture (Example for 3 MHz BW)*

### **3.4 Handling Complex-Valued Signals within Real-Valued Neural Network Frameworks**

Standard deep learning frameworks like PyTorch, along with their rich ecosystem of layers and optimizers, are predominantly designed to operate on real-valued tensors. However, the signals in a wireless communication system are fundamentally complex-valued. A robust and widely adopted method is required to bridge this gap.

The standard approach is to represent a complex number $ z \= a \+ jb $ as a real-valued vector of length two, $ \[a, b\] $. This is extended to tensors by adding an extra dimension of size two to hold the real (I) and imaginary (Q) parts separately.3 For example, a batch of complex OFDM grids with shape

\[batch\_size, num\_subcarriers, num\_symbols\] would be converted into a real-valued tensor of shape \[batch\_size, num\_subcarriers, num\_symbols, 2\].

This representation is applied at the interface between the communication-specific layers and the neural network layers:

* **At the Decoder Input:** The complex-valued grid $ \\mathbf{Y} $ received from the OFDM demodulator is split into its real and imaginary parts to form the two-channel input to the first convolutional layer of the decoder.  
* **At the Encoder Output:** The final convolutional layer of the encoder is configured to have two output channels. These are interpreted as the real and imaginary parts of the output grid $ \\mathbf{X} $. They are then combined to form a complex tensor before being passed to the power normalization layer and the OFDM modulator.

While research into true complex-valued neural networks (CVNNs) exists, which define operations like convolution and activation functions directly in the complex domain, the two-channel real-valued representation is the most practical and well-supported method for this project.31 It allows for the direct use of highly optimized, off-the-shelf layers (e.g.,

Conv2d, BatchNorm2d, ReLU) without requiring custom implementations, significantly simplifying the development process while achieving state-of-the-art performance.

## **IV. A Robust Training and Convergence Strategy**

### **4.1 Loss Function Selection and Optimization for Bit-Level Recovery**

The choice of loss function is paramount as it defines the objective that the end-to-end system is trained to optimize. Given that the ultimate goal of a digital communication system is the accurate recovery of the transmitted information bits, the problem is best framed as a multi-label binary classification task. For each of the $ K $ bits in the input vector $ \\mathbf{b} $, the decoder outputs a probability $ \\hat{b}\_i \\in $ that the $ i $-th bit is a '1'.

The most suitable loss function for this task is the **Binary Cross-Entropy (BCE)** loss, averaged over all $ K $ bits in the block.6 The BCE loss for a single bit $ b\_i $ and its predicted probability $ \\hat{b}

i $ is given by:  
$$L{BCE}(b\_i, \\hat{b}\_i) \= \-\[b\_i \\log(\\hat{b}i) \+ (1 \- b\_i) \\log(1 \- \\hat{b}i)\]$$The total loss for the block is the mean of the individual bit losses:$$L(\\mathbf{b}, \\hat{\\mathbf{b}}) \= \\frac{1}{K} \\sum{i=1}^{K} L{BCE}(b\_i, \\hat{b}\_i)$$  
This loss function directly penalizes the model for incorrect bit predictions. By optimizing to minimize BCE, the training process is explicitly guided towards maximizing bit-level accuracy. This is a more direct and effective approach than optimizing for an intermediate metric, such as the Mean Squared Error (MSE) between the transmitted and received constellation symbols. Optimizing for MSE would prioritize signal fidelity, which does not always correlate perfectly with bit error rate. The BCE loss allows the autoencoder the freedom to learn highly distorted or non-standard constellations if such representations prove to be more separable and robust at the receiver after passing through the channel, ultimately leading to a lower BER.  
For optimization, the **Adam optimizer** is the recommended choice. It is an adaptive learning rate optimization algorithm that is computationally efficient, has low memory requirements, and is well-suited for problems that are large in terms of data and/or parameters. It combines the advantages of two other popular extensions of SGD: AdaGrad and RMSProp, making it a robust default choice for training deep neural networks.6

### **4.2 A Curriculum Learning Approach for Progressive Model Training**

Training a complex, deep autoencoder on a multifaceted task involving multiple simultaneous channel impairments can be challenging and unstable. The optimization landscape is highly non-convex, and attempting to learn everything at once can lead to poor local minima or slow convergence. A more effective strategy is **curriculum learning**, where the model is trained on progressively more difficult tasks, using the weights from the simpler task as initialization for the next, more complex one.

A structured training curriculum for this project is proposed as follows:

1. **Phase 1: AWGN Channel at High SNR.** The initial training phase should use the simplest possible channel: AWGN only, at a relatively high SNR (e.g., 20 dB). The goal of this phase is for the autoencoder to learn the fundamental mapping from bits to a valid, power-constrained resource grid and back. It establishes a baseline encoding/decoding scheme without the complexities of fading or phase errors.  
2. **Phase 2: Training Across a Range of SNRs.** Once the model has converged on the high-SNR AWGN channel, it is fine-tuned on data generated over a wide range of SNRs (e.g., 0 dB to 20 dB, sampled uniformly).10 This step is crucial for generalization. It forces the encoder to learn a single, robust representation that performs well in both noise-limited (low SNR) and interference-limited (high SNR) regimes. The network must learn to balance the need for large Euclidean distance between constellation points (for low SNR) with the potential for higher spectral efficiency (at high SNR). This process of training across SNRs is a form of implicit robustness training.  
3. **Phase 3: Introducing Rayleigh Fading.** Using the SNR-robust model from Phase 2 as a starting point, the Rayleigh fading channel layer is introduced. The model is then fine-tuned on this more complex channel. Because the network has already learned a robust mapping for AWGN, it can now focus its learning capacity on adapting this mapping to be resilient to the amplitude and phase variations introduced by frequency-selective fading.  
4. **Phase 4: Introducing Residual CFO.** In the final phase, the differentiable residual CFO layer is added to the channel model. The model, now robust to AWGN and fading, is fine-tuned one last time to also become invariant to the time-varying phase rotations caused by frequency errors.

This staged approach simplifies the optimization problem at each step, leading to more stable training, faster convergence, and ultimately, a more robust and better-performing final model.

### **4.3 Techniques for Monitoring and Ensuring Model Convergence**

Effective monitoring during the training process is essential for understanding model behavior, diagnosing problems, and ensuring that the final model is the best possible one. Several techniques should be employed:

* **Metrics Tracking:** During training and validation, multiple metrics should be tracked and logged.  
  * **Loss:** The primary training objective, the BCE loss, should be monitored on both the training and validation sets. A decreasing training loss indicates the model is learning, while a diverging or plateauing validation loss can signal overfitting.  
  * **Bit Error Rate (BER) / Block Error Rate (BLER):** Since BER is the ultimate performance metric for the communication system, it should be calculated on the validation set at the end of each epoch. The model that achieves the lowest validation BER should be saved as the final candidate model. BLER (where a block is one 6-symbol slot) is also a valuable metric.  
* **Visualization Tools:** Visualizing the internal state and outputs of the network provides invaluable qualitative insights.  
  * **Loss and BER Curves:** Plotting the training and validation loss/BER curves over epochs is the most fundamental way to assess convergence, stability, and overfitting. Tools like TensorBoard are excellent for this.  
  * **Learned Constellations:** At regular intervals during training, the output of the encoder for a fixed set of input bit patterns can be plotted as a 2D scatter plot of I/Q values.3 Observing how these constellations evolve—from a random cloud to a structured, well-separated geometry—provides a clear indication of the transmitter's learning progress.  
* **Standard PyTorch Training Practices:**  
  * **TensorBoard Integration:** PyTorch integrates seamlessly with TensorBoard via torch.utils.tensorboard.SummaryWriter. This allows for real-time logging and visualization of all specified metrics during training.  
  * **Model Checkpointing:** A common practice in PyTorch is to manually save the model's state dictionary (model.state\_dict()) at the end of each epoch or whenever the validation performance (e.g., validation BER) improves. This ensures that the best-performing model is preserved.  
  * **Early Stopping:** This can be implemented within the training loop by monitoring the validation loss or BER and stopping the training if no improvement is observed for a specified number of "patience" epochs. This prevents wasted computation and helps mitigate overfitting.

### **4.4 Hyperparameter Tuning and Validation Strategy**

The performance of a deep learning model is highly dependent on its hyperparameters. While the proposed architecture provides a strong starting point, some tuning will likely be necessary to achieve optimal performance. Key hyperparameters include:

* **Learning Rate:** The step size for the Adam optimizer. A value that is too high can lead to instability, while a value that is too low can result in extremely slow convergence. A typical starting point is $ 10^{-3} $ or $ 10^{-4} $.  
* **Batch Size:** The number of training examples used in one iteration of gradient descent. Larger batch sizes provide more stable gradient estimates but require more memory. Typical values range from 32 to 256\.  
* **Network Depth and Width:** The number of convolutional layers and the number of filters in each layer. A deeper or wider network has more capacity to learn complex functions but is also more prone to overfitting and computationally expensive.

A rigorous approach to training requires a clean separation of data:

1. **Training Set:** A large dataset used to compute gradients and update the network weights.  
2. **Validation Set:** A separate, smaller dataset used to evaluate the model's performance after each training epoch. This set is used for hyperparameter tuning and for model selection via checkpointing. It provides an unbiased estimate of the model's performance on data it has not been trained on.  
3. **Test Set:** A final, held-out dataset that is used only once, after all training and hyperparameter tuning is complete. The performance on this set (e.g., the final BER vs. SNR curve) is the result that should be reported as the true, unbiased performance of the final system. This strict separation is critical to avoid "data leakage" and ensure academic and scientific rigor.

## **V. A Phased Implementation Guide Using PyTorch**

This section provides a practical, step-by-step guide to implementing the proposed OFDM autoencoder using PyTorch. PyTorch is an excellent framework for this task, offering a dynamic computation graph and a rich ecosystem that provides the flexibility needed for custom physical layer modeling.33 The implementation will follow the curriculum learning strategy, building the system from a simple baseline to the full, robust model by creating custom, differentiable

torch.nn.Module classes for each component.

### **5.1 Environment Setup: Configuring PyTorch and GPU Acceleration**

A properly configured Python environment is the first step for a successful implementation.

1. **Python Environment:** Use a virtual environment (e.g., venv or conda) to manage project dependencies and avoid conflicts.  
2. **PyTorch with GPU Support:** Install PyTorch by following the official instructions on the PyTorch website. Be sure to select the appropriate version for your system's CUDA toolkit to enable GPU acceleration, which is critical for training performance.  
3. **Verification:** Confirm that PyTorch is installed correctly and can detect your GPU by running the following Python script:  
   Python  
   import torch  
   print("PyTorch Version:", torch.\_\_version\_\_)  
   print("CUDA Available:", torch.cuda.is\_available())  
   if torch.cuda.is\_available():  
       print("CUDA Version:", torch.version.cuda)  
       print("Device Name:", torch.cuda.get\_device\_name(0))

   A successful run will print the library versions and confirm CUDA availability.

| Phase | Objective | Bandwidth | Channel Components | Key PyTorch Modules/Functions | Expected Outcome |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **1** | Establish a baseline and learn the fundamental encoding/decoding mapping. | 3 MHz | AWGN | torch.nn.Module, torch.fft.ifft, torch.fft.fft, torch.nn.ConvTranspose2d, torch.nn.Conv2d | Low BER ($ \< 10^{-5} )athighSNR( \> 15 $ dB). Well-structured learned constellation. |
| **2** | Generalize the model to be robust to frequency-selective fading. | 3 MHz | AWGN \+ Rayleigh Fading | Custom TDL Fading Module, torch.nn.functional.conv1d | Convergence to a low BER, albeit at a higher required SNR compared to Phase 1\. |
| **3** | Make the model robust to residual frequency errors. | 3 MHz | AWGN \+ Rayleigh \+ CFO | Custom Differentiable CFO Module | Stable, low BER performance even with random CFO applied to each batch. |
| **4** | Scale the model to the full system specification and perform final benchmarking. | 20 MHz | AWGN \+ Rayleigh \+ CFO | All of the above, with updated parameters. | A final, robust model operating at 20 MHz. BER vs. SNR curve outperforming a conventional baseline. |

*Table 4: Phased Implementation and Training Plan with PyTorch*

### **5.2 Phase 1: Baseline Implementation (3MHz Bandwidth, AWGN Channel)**

This phase focuses on building and training the core autoencoder with the simplest channel model to verify the architecture and data flow.

1\. Defining System Parameters and Helper Modules:  
Define the 3 MHz system parameters in a configuration file or script. Then, create custom torch.nn.Module classes for the fixed OFDM processing blocks.

Python

import torch  
import torch.nn as nn

class OFDMModulator(nn.Module):  
    def \_\_init\_\_(self, fft\_size, cp\_len, num\_sc):  
        super().\_\_init\_\_()  
        self.fft\_size \= fft\_size  
        self.cp\_len \= cp\_len  
        self.num\_sc \= num\_sc

    def forward(self, x\_grid): \# x\_grid: \[batch, num\_sym, num\_sc\] complex  
        \# Map to FFT bins  
        batch\_size, num\_sym, \_ \= x\_grid.shape  
        x\_fft \= torch.zeros(batch\_size, num\_sym, self.fft\_size, dtype=x\_grid.dtype, device=x\_grid.device)  
        start\_idx \= (self.fft\_size \- self.num\_sc) // 2  
        x\_fft\[:, :, start\_idx : start\_idx \+ self.num\_sc\] \= x\_grid  
          
        \# IFFT  
        x\_time \= torch.fft.ifft(x\_fft, n=self.fft\_size, dim=-1)  
          
        \# Add Cyclic Prefix  
        cp \= x\_time\[:, :, \-self.cp\_len:\]  
        x\_time\_cp \= torch.cat(\[cp, x\_time\], dim=-1)  
          
        return x\_time\_cp.view(batch\_size, \-1) \# Serialize

class OFDMDemodulator(nn.Module):  
    \#... Symmetrical implementation with CP removal and FFT...

2\. Implementing the End-to-End Model:  
Assemble the encoder, decoder, and the custom OFDM modules into a single end-to-end model.

Python

class OFDMAutoencoder(nn.Module):  
    def \_\_init\_\_(self, encoder, decoder, modulator, demodulator):  
        super().\_\_init\_\_()  
        self.encoder \= encoder  
        self.decoder \= decoder  
        self.modulator \= modulator  
        self.demodulator \= demodulator

    def forward(self, b, ebno\_db): \# b: \[batch, K\]  
        \# Transmitter  
        x\_grid\_real \= self.encoder(b) \# Output: \[batch, num\_sym, num\_sc, 2\]  
        x\_grid \= torch.complex(x\_grid\_real\[..., 0\], x\_grid\_real\[..., 1\])  
          
        \# Power Normalization  
        pwr \= torch.mean(torch.abs(x\_grid)\*\*2, dim=(-1, \-2), keepdim=True)  
        x\_normalized \= x\_grid / torch.sqrt(pwr)  
          
        x\_time \= self.modulator(x\_normalized)  
          
        \# AWGN Channel  
        noise\_std \= (10\*\*(-ebno\_db / 10.0)).sqrt()  
        noise \= torch.randn\_like(x\_time, dtype=torch.cfloat) \* (noise\_std / 1.414)  
        y\_time \= x\_time \+ noise  
          
        \# Receiver  
        y\_grid \= self.demodulator(y\_time)  
        y\_grid\_real \= torch.stack(\[y\_grid.real, y\_grid.imag\], dim=-1)  
        b\_hat \= self.decoder(y\_grid\_real)  
        return b\_hat

3\. Training and Evaluation:  
Write a standard PyTorch training loop. Generate random bits for each batch, calculate the BCE loss using torch.nn.BCELoss, and update the model weights using an Adam optimizer. Monitor performance by calculating the BER on a separate validation set.

### **5.3 Phase 2: Incorporating Frequency-Selective Fading**

This phase introduces a more realistic channel by implementing a differentiable Tapped Delay Line (TDL) fading model.

1\. Implementing the Fading Channel Module:  
Create a custom nn.Module that simulates a Rayleigh fading channel. This involves generating random complex Gaussian taps for a channel impulse response (CIR) and convolving it with the time-domain signal.

Python

import torch.nn.functional as F

class TDLChannel(nn.Module):  
    def \_\_init\_\_(self, num\_taps):  
        super().\_\_init\_\_()  
        self.num\_taps \= num\_taps

    def forward(self, x\_time): \# x\_time: \[batch, num\_samples\] complex  
        batch\_size \= x\_time.shape  
          
        \# Generate random channel taps for each example in the batch  
        \# Taps are complex Gaussian  
        taps\_real \= torch.randn(batch\_size, 1, self.num\_taps, device=x\_time.device)  
        taps\_imag \= torch.randn(batch\_size, 1, self.num\_taps, device=x\_time.device)  
        taps \= torch.complex(taps\_real, taps\_imag) \* (1 / 1.414)  
          
        \# Apply channel via 1D convolution  
        \# PyTorch conv1d requires real tensors, so we convolve I and Q components separately  
        x\_real, x\_imag \= x\_time.real.unsqueeze(1), x\_time.imag.unsqueeze(1)  
        taps\_real, taps\_imag \= taps.real, taps.imag  
          
        y\_real \= F.conv1d(x\_real, taps\_real, padding='same') \- F.conv1d(x\_imag, taps\_imag, padding='same')  
        y\_imag \= F.conv1d(x\_real, taps\_imag, padding='same') \+ F.conv1d(x\_imag, taps\_real, padding='same')  
          
        return torch.complex(y\_real.squeeze(1), y\_imag.squeeze(1))

2\. Integration and Retraining:  
Insert an instance of the TDLChannel into the forward pass of the OFDMAutoencoder model, after the modulator and before the AWGN is added. Load the best weights from Phase 1 and fine-tune the model on this more complex channel environment.

### **5.4 Phase 3: Building Robustness to Frequency Errors**

The final channel impairment, residual CFO, is added in this phase.

1\. Implementing a Differentiable CFO Module:  
Create a custom nn.Module for applying CFO. This module will be placed in the time domain, before the fading channel.

Python

class ResidualCFO(nn.Module):  
    def \_\_init\_\_(self, max\_cfo\_norm, sampling\_freq, subcarrier\_spacing):  
        super().\_\_init\_\_()  
        self.max\_cfo\_norm \= max\_cfo\_norm  
        self.sampling\_freq \= sampling\_freq  
        self.subcarrier\_spacing \= subcarrier\_spacing

    def forward(self, x\_time): \# x\_time: \[batch, num\_samples\] complex  
        batch\_size, num\_samples \= x\_time.shape  
          
        \# Sample random CFO for each example in the batch  
        cfo\_norm \= (torch.rand(batch\_size, 1, device=x\_time.device) \* 2 \- 1) \* self.max\_cfo\_norm  
        delta\_f \= cfo\_norm \* self.subcarrier\_spacing  
          
        \# Create time vector and phase rotation  
        t \= torch.arange(0, num\_samples, device=x\_time.device) / self.sampling\_freq  
        phase\_rotation \= 2.0 \* torch.pi \* delta\_f \* t  
          
        cfo\_phasor \= torch.exp(1j \* phase\_rotation)  
        return x\_time \* cfo\_phasor

2\. Integration and Training:  
Insert the ResidualCFO module into the OFDMAutoencoder's forward pass. Load the weights from Phase 2 and perform the final fine-tuning step. The model now learns to be robust to all three impairments simultaneously.

### **5.5 Phase 4: Scaling to Full System Specification (20MHz Bandwidth)**

The final phase involves scaling the proven 3 MHz design to the full 20 MHz specification and conducting a final benchmark.

1\. Adapting the Architecture:  
Update the system parameters for the 20 MHz bandwidth (e.g., $ N\_{sc}=1200, N\_{FFT}=2048, N\_{cp}=144 $). The increased dimensions of the resource grid will require scaling the CNN architectures. This typically involves adding more convolutional/transposed convolutional layers and potentially increasing the number of filters (channels) in each layer to ensure the model has sufficient capacity to handle the larger input and learn the more complex mapping.  
2\. Final Training and Benchmarking:  
Instantiate a new model with the scaled 20 MHz architecture. A full end-to-end training run is performed following the curriculum learning approach (or fine-tuning from the 3 MHz model if the architectures are sufficiently similar).  
The final step is to produce a comprehensive BER vs. $ E\_b/N\_0 $ performance curve. This evaluation must be done on a held-out test set. For a meaningful comparison, this curve should be plotted alongside a conventional baseline system implemented in PyTorch. A suitable baseline would consist of:

* Standard QAM modulation (e.g., 16-QAM).  
* A standard channel code (e.g., a simple convolutional code or a more advanced LDPC code if a suitable PyTorch library is available).  
* A pilot-based channel estimator (e.g., Least Squares) followed by an LMMSE equalizer.

The expected outcome is that the end-to-end learned autoencoder will demonstrate superior performance compared to the conventional, block-based baseline, validating the benefits of the data-driven, joint optimization approach.

## **VI. Conclusion**

This report has outlined a comprehensive, PhD-level plan for the design and implementation of a point-to-point OFDM modem for drone communications, leveraging a deep learning autoencoder. The proposed approach replaces a significant portion of the conventional receiver chain with a single, end-to-end optimized neural network, promising substantial performance gains and robustness.

The core of the proposed solution is a **Convolutional Autoencoder**, an architecture uniquely suited to the 2D time-frequency structure of the OFDM resource grid. The CNN-based encoder learns a novel, spatially-coupled modulation scheme that spreads information across time and frequency, while the CNN-based decoder performs holistic, joint channel estimation, equalization, and demapping. This integrated approach avoids the suboptimalities of traditional block-based designs and can adapt to complex channel impairments that are difficult to address with analytical models.

A critical component of the design is the use of a **differentiable channel model**, which enables end-to-end training by allowing gradients to flow from the receiver's output back to the transmitter's input. The proposed model incorporates AWGN, frequency-selective Rayleigh fading, and residual Carrier Frequency Offset, ensuring the trained system is robust to a realistic set of wireless channel impairments.

To ensure successful implementation, a **phased development plan** using PyTorch has been detailed. This plan follows a curriculum learning strategy, starting with a simplified 3 MHz AWGN channel and progressively adding complexity, scaling up to the full 20 MHz specification with all channel effects. This methodical approach stabilizes training, simplifies debugging, and ensures the development of a robust final model.

The successful execution of this plan will result in a novel OFDM modem that not only meets the specified requirements but also serves as a powerful demonstration of the potential of AI-native physical layers. By combining the known-optimal structure of OFDM with the powerful learning capabilities of deep neural networks, this work stands at the forefront of research into next-generation wireless systems, paving the way for the intelligent, adaptive, and high-performance communication networks of the future.

#### **Works cited**

1. Channel model for end-to-end learning of communications systems: A survey, accessed August 17, 2025, [https://www.researchgate.net/publication/359864696\_Channel\_model\_for\_end-to-end\_learning\_of\_communications\_systems\_A\_survey](https://www.researchgate.net/publication/359864696_Channel_model_for_end-to-end_learning_of_communications_systems_A_survey)  
2. Deep Learning-Based Communication Over the Air \- arXiv, accessed August 17, 2025, [https://arxiv.org/pdf/1707.03384](https://arxiv.org/pdf/1707.03384)  
3. OFDM-Autoencoder for End-to-End Learning of Communications ..., accessed August 17, 2025, [https://arxiv.org/pdf/1803.05815](https://arxiv.org/pdf/1803.05815)  
4. Deep Learning in Physical Layer Communications \- arXiv, accessed August 17, 2025, [https://arxiv.org/pdf/1807.11713](https://arxiv.org/pdf/1807.11713)  
5. Deep Learning in Physical Layer Communications, accessed August 17, 2025, [https://www.cse.wustl.edu/\~jain/cse574-22/ftp/dl\_phy/index.html](https://www.cse.wustl.edu/~jain/cse574-22/ftp/dl_phy/index.html)  
6. End-to-End Autoencoder Communications with Optimized ... \- arXiv, accessed August 17, 2025, [https://arxiv.org/pdf/2201.01388](https://arxiv.org/pdf/2201.01388)  
7. Deep learning in physical layer communications: Evolution and prospects in 5G and 6G networks \- ResearchGate, accessed August 17, 2025, [https://www.researchgate.net/publication/373092570\_Deep\_learning\_in\_physical\_layer\_communications\_Evolution\_and\_prospects\_in\_5G\_and\_6G\_networks](https://www.researchgate.net/publication/373092570_Deep_learning_in_physical_layer_communications_Evolution_and_prospects_in_5G_and_6G_networks)  
8. End-to-End Deep Learning in Phase Noisy Coherent Optical Link \- arXiv, accessed August 17, 2025, [https://arxiv.org/pdf/2502.21209](https://arxiv.org/pdf/2502.21209)  
9. Model-Driven Deep Learning for Physical Layer Communications \- arXiv, accessed August 17, 2025, [https://arxiv.org/pdf/1809.06059](https://arxiv.org/pdf/1809.06059)  
10. Performance Comparison of Autoencoder based OFDM Communication System with Wi-Fi, accessed August 17, 2025, [http://paper.ijcsns.org/07\_book/202305/20230519.pdf](http://paper.ijcsns.org/07_book/202305/20230519.pdf)  
11. \[2502.21209\] End-to-End Deep Learning in Phase Noisy Coherent Optical Link \- arXiv, accessed August 17, 2025, [https://arxiv.org/abs/2502.21209](https://arxiv.org/abs/2502.21209)  
12. Channel Model for End-To-End Learning of Communications Systems- Survey \- Scribd, accessed August 17, 2025, [https://www.scribd.com/document/898733174/Channel-Model-for-End-To-End-Learning-of-Communications-Systems-Survey](https://www.scribd.com/document/898733174/Channel-Model-for-End-To-End-Learning-of-Communications-Systems-Survey)  
13. Differentiable Channels Are All You Need \- OSTI, accessed August 17, 2025, [https://www.osti.gov/servlets/purl/1881079](https://www.osti.gov/servlets/purl/1881079)  
14. End-to-End Learning of Communications Systems Without a Channel Model \- arXiv, accessed August 17, 2025, [https://arxiv.org/pdf/1804.02276](https://arxiv.org/pdf/1804.02276)  
15. arxiv.org, accessed August 17, 2025, [https://arxiv.org/html/2506.17530v1](https://arxiv.org/html/2506.17530v1)  
16. Deep-OFDM: Robust Neural Modulation for High Mobility \- OpenReview, accessed August 17, 2025, [https://openreview.net/pdf?id=UZsVnubmtC](https://openreview.net/pdf?id=UZsVnubmtC)  
17. \[1803.05815\] OFDM-Autoencoder for End-to-End Learning of Communications Systems, accessed August 17, 2025, [https://arxiv.org/abs/1803.05815](https://arxiv.org/abs/1803.05815)  
18. IJCSNS \- International Journal of Computer Science and Network Security, accessed August 17, 2025, [http://ijcsns.org/07\_book/html/202305/202305019.html](http://ijcsns.org/07_book/html/202305/202305019.html)  
19. Deep-OFDM: Neural Modulation for High Mobility \- arXiv, accessed August 17, 2025, [https://arxiv.org/pdf/2506.17530](https://arxiv.org/pdf/2506.17530)  
20. \[2506.17530\] Deep-OFDM: Neural Modulation for High Mobility \- arXiv, accessed August 17, 2025, [https://arxiv.org/abs/2506.17530](https://arxiv.org/abs/2506.17530)  
21. OFDM Autoencoder for Wireless Communications \- MathWorks, accessed August 17, 2025, [https://www.mathworks.com/help/comm/ug/ofdm-autoencoder-wireless-communications.html](https://www.mathworks.com/help/comm/ug/ofdm-autoencoder-wireless-communications.html)  
22. Rayleigh fading channels in mobile digital communication systems Part I: Characterization, accessed August 17, 2025, [https://www.researchgate.net/publication/3195748\_Rayleigh\_fading\_channels\_in\_mobile\_digital\_communication\_systems\_Part\_I\_Characterization](https://www.researchgate.net/publication/3195748_Rayleigh_fading_channels_in_mobile_digital_communication_systems_Part_I_Characterization)  
23. Multipath Fading | PySDR: A Guide to SDR and DSP using Python, accessed August 17, 2025, [https://pysdr.org/content/multipath\_fading.html](https://pysdr.org/content/multipath_fading.html)  
24. What is Carrier Frequency Offset (CFO) and How It Distorts the Rx Symbols | Wireless Pi, accessed August 17, 2025, [https://wirelesspi.com/what-is-carrier-frequency-offset-cfo-and-how-it-distorts-the-rx-symbols/](https://wirelesspi.com/what-is-carrier-frequency-offset-cfo-and-how-it-distorts-the-rx-symbols/)  
25. Deep Learning Based OFDM Physical-Layer Receiver for Extreme Mobility \- Tampere University Research Portal, accessed August 17, 2025, [https://researchportal.tuni.fi/files/62991316/ML\_PHY\_Receiver\_for\_Extreme\_Mobility\_Asilomar\_2021.pdf](https://researchportal.tuni.fi/files/62991316/ML_PHY_Receiver_for_Extreme_Mobility_Asilomar_2021.pdf)  
26. Deep Learning for Channel Estimation in Physical Layer Wireless Communications: Fundamental, Methods, and Challenges \- MDPI, accessed August 17, 2025, [https://www.mdpi.com/2079-9292/12/24/4965](https://www.mdpi.com/2079-9292/12/24/4965)  
27. Convolutional Neural Network Auto Encoder Channel Estimation Algorithm in MIMO-OFDM System \- ResearchGate, accessed August 17, 2025, [https://www.researchgate.net/publication/357489746\_Convolutional\_Neural\_Network\_Auto\_Encoder\_Channel\_Estimation\_Algorithm\_in\_MIMO-OFDM\_System](https://www.researchgate.net/publication/357489746_Convolutional_Neural_Network_Auto_Encoder_Channel_Estimation_Algorithm_in_MIMO-OFDM_System)  
28. Convolutional Neural Network Auto Encoder Channel Estimation Algorithm in MIMO-OFDM System \- Semantic Scholar, accessed August 17, 2025, [https://pdfs.semanticscholar.org/5788/e00917dded31b982f69609b759ed636eec3f.pdf](https://pdfs.semanticscholar.org/5788/e00917dded31b982f69609b759ed636eec3f.pdf)  
29. Tutorial 9: Deep Autoencoders — UvA DL Notebooks v1.2 documentation, accessed August 17, 2025, [https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial\_notebooks/tutorial9/AE\_CIFAR10.html](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html)  
30. Deep Reservoir Computing Meets 5G MIMO-OFDM Systems in Symbol Detection, accessed August 17, 2025, [https://ojs.aaai.org/index.php/AAAI/article/view/5481/5337](https://ojs.aaai.org/index.php/AAAI/article/view/5481/5337)  
31. ONLINE ML-BASED JOINT CHANNEL ESTIMATION AND MIMO DECODING FOR DYNAMIC CHANNELS \- Biblioteca da SBrT, accessed August 17, 2025, [https://biblioteca.sbrt.org.br/articlefile/4709.pdf](https://biblioteca.sbrt.org.br/articlefile/4709.pdf)  
32. A Learned OFDM Receiver Based on Deep Complex-Valued Convolutional Networks \- NSF-PAR, accessed August 17, 2025, [https://par.nsf.gov/servlets/purl/10393844](https://par.nsf.gov/servlets/purl/10393844)  
33. Tensors and Dynamic neural networks in Python with strong GPU acceleration \- GitHub, accessed August 17, 2025, [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)