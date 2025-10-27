Developed a Generative Pre-Trained Transformer model with 12 layers, 12 attention heads, and 768-dimensional embeddings, replicating the GPT-355M architecture.
Implemented Multi-head self-attention, Layer Normalization, and Feed-Forward layers from scratch, improving model scalability by 30% in experiments compared to a baseline

Built a tokenizer handling 50,257 tokens using OpenAI’s tiktoken library for efficient text processing also called as Byte-Pair Encoding.
Optimized training using PyTorch DataLoader and GPU acceleration, reducing training time per epoch by 40%.

Implemented a complete model training and evaluation framework, successfully initializing the custom architecture by loading pre-trained GPT-2 weights and Fine-Tuned the model using the AdamW optimizer.

Intro:
I’ll just start with a quick intro of what a GPT is?. Generative Pre-trained Transformative. You all know whats a Generative and Pre-Trained the name itself is saying it, ever looked into whats a Transformative is?. Its a Architecture that act’s like an Engine to all the LLM’s. 
Still wondering ? What it is. I’ve got you. We have Total of 3 -Stages. 

Stage-1: Building the Model
    Data-Preprocessing & Sampling:
	Steps:
	1. Give your input Ex. The cat next to the dog, jumped!
	2. Convert it into Token’s called as Token Embeddings.  (Method: Byte-Pair Encoding -  using “tiktoken”) - this will tell where the token 
	3. Now, Do the Positional Embeddings.  (This will tell whats in the token)
	4. After this, Input Embeddings = Token Embeddings + Positional Embeddings.
	5. Do the Dropout, basically there will be some lazy neurons, so the dropout will randomly turnoff neurons, to work effectively. 
        So, we have loaded the Data (Dropout Input Embeddings). Now, What happeness? This will 	pass to  the Transfomers Block! Yes, that’s were the Transformative Comes in. 
   Transformers:
	Steps:
	1. Do,  Layer Normalization. (Stability in Neural Network) we calculate mean & var. 
	2. Now, calculate Masked Multi-head Attention- Basically,  the work is to give the Context Vector(Which gives the semantic meaning how each word related to each other. Our Ex. Cat and dog are animals, here the cat is jumped that’s what context vector will say).
	3. Next step, Dropout - Same as the previous one. 
	4.  Shortcut Connection:   Dropout  for Context vector(step3.) + (Dropout Input Embeddings)
	5. Now, Again Do Layer Normalization. 
	6. Next step is, FeedForward -
		i. Apply Linear Layer (Multiply 4 times the config= Ex. 4 * 768Dim  for input Dimension) - for Expansion
		 ii. Apple Gaussian Error Linear Unit (GELU) Activation Function[Refer online for the Formula]
		iii. Apply Linear Layer (Multiply 4 times the config= Ex. 4 * 768Dim ) now for Contraction. (Back to 768 Dim)
	7. Next step, Dropout - Same as the previous one. 
	8. Shortcut Connection: previous_shortcut + current dropout
      This ShortConnection is the Transformers Block Output. 

 Final Steps:
         1. Do, Layer Normalization (Final Layer Normalization) - Same as Above.
         2.  Identify the Logits Which is the final predicted Next Tokens. Decode them back to 	words.. this is will be your final predicted word. 
Remember, this is just one Transformer Block, We will use 12 of these in our models.

Stage-2:	
	Pre-Training: Now that we have build our model and predicted next word,  it will be random. Because we haven’t trained out model yet. Now, we are gonna do that. 
	Steps: 
	    1. From the output logins, To remove the randomness, we have 2 techniques (Temperature Scaling & Top-k Sampling). Combine together 
	    2. Then, you load the pre-trained weights from GPT-2 (355M) and add to your model. 


Stage-3:
	FineTuning: This is the last step. You can fine-tune your model in 2 ways. Instruction Fine-Tuning and 2. Classification Fine-tuning.  
		Instruction  Fine-Tuning: You can add instructions to your model. 
			Ex. “You have won a lottery”
			   Instruction: Is the above text spam or not
					Answer with yes or no 
	Classification Fine-tuning: It works basically like have only two output types of responses.  Either Yes or No. 

