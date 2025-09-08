# Text-To-Image-Diffusion-Transformer
- Implementation of each block from scratch, below is the representation of the Architecture.
<img src="Images/Architecture.png" width="400" height="450"/>
- PixArt-Alpha Block
<img src="Images/t2i.png" width="400" height="450"/>

## My Architecture
- Step 1, I won't use a single Text Encodings, I would use multiple Text Encodings, so as to understand the context of text is richer. It is evident that the reacher your Text Embeddings are better is the Generated Image from it (SD3 uses multiple Text Encoders).
- I am using GPT-Neo, T5, and CLIP
- Concatenated them, and added a output later of embedding dimension 
- Outputs the Text Embeddings


- Step 2, Image Encodings
- Used SDE3 implemented VAE Image Encoder
- Dependency issue while running up DCAE
- I am able to execute the DCAE Implementation as well


- Ste 3, Noise addition to Latents
- Implementeation of Noise at random timestep t, using DDPM along with cosine beta scheduler
- Also Implement the Time Embedding Block, with Sinuodial Time Steps so as to include phase information
- for a Linear batch of Time t, it outputs (batch, Embed Dim) here my Embed Dim is 768


- Step 4, Patchify
- The Noised latents are converted in to patches
- Patches after flattening are added to positional Encodings
- These both are added up and the resultant is sent to input Layer of DiT block
- Along with patchification layer, also add Unpatchify dunctions, reversing it with ConvTranspose2D.


- Step 5, Encoder Block
- This block is cummulative of all the steps mentioned above before sending the noised patchified latents
to Input in to Transformer block.
- All the methods are cummulates in Encoder Class.


- Step 6, Main DiT Blocks
- Scale Block and Scale Shift block, this are blocks which is intaken by the shared parameters which is 
chunked from the Time Embedding Vector.
- The other Most important blocks are MultiHead Self Attention which is a projection of image latents and itself, 
whereas the other block MultiHead Cross Attention is a projection of Text Embedding and Image latents.
- The blocks are something like ScaleShift -> MultiHead Self Attention -> Scale -> Multi Head Cross Attention ->
ScaleShift -> Feed Forward Block -> Scale.. this entire blocks are repeated for N times
- There are also Residual connections in between the blocks
- The Entire DiT Block is encapsulated in NDiTBlock class

