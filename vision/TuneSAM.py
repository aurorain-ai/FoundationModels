import torch
import torch.nn.functional as F
from torchvision.transforms import Resize
from SAM import SamPredictor, sam_model_registry, utils

# Load the SAM model
sam_model = sam_model_registry['vit_b'](checkpoint='sam_vit_b_01ec64.pth')

# Set up optimizer
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters())

# Set up loss function
loss_fn = torch.nn.MSELoss()

# Load custom dataset
# Here, we should replace 'custom_dataset' with the dataset loader
custom_dataset = ...

# Set device, better to use A100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training loop
for epoch in range(num_epochs):  # num_epochs to be defined
    for input_image, box_torch, gt_binary_mask in custom_dataset:
        input_image, box_torch, gt_binary_mask = input_image.to(device), box_torch.to(device), gt_binary_mask.to(device)

        # Image encoding
        with torch.no_grad():
            image_embedding = sam_model.image_encoder(input_image)
        
        # Prompt encoding
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(points=None, boxes=box_torch, masks=None)

        # Mask decoding
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Postprocessing
        upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)

        # Generate binary mask
        binary_mask = F.normalize(F.threshold(upscaled_masks, 0.0, 0)).to(device)

        # Calculate loss and backpropagate
        loss = loss_fn(binary_mask, gt_binary_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save the fine-tuned model
torch.save(sam_model.state_dict(), 'fine_tuned_sam.pth')
