import click
import yaml
import tqdm

import torch
import torchvision

from datasets import get_dataset
from models import get_model_transforms


from attribution import (
    get_composite,
)

from models import get_model_transforms


def initialize_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def lrp_attributor(model, inputs, targets, composite, device, **kwargs):
    """
    Compute the relevance using LRP

    Args:
        model (torch.nn.module): the model to be explained
        inputs (torch.tensor): inputs or the given images
        targets (torch.tensor): targets of the given images
        composite (): lrp composite
        device (): device to be used

    Returns:
        (torch.tensor): the computed heatmap using LRP
    """
    with torch.enable_grad():
        composite = composite

        inputs.requires_grad = True
        with composite.context(model) as modified_model:
            # If kwargs exists
            if bool(kwargs):
                if kwargs["method"] in ["latent_flipping", "latent_insertion"]:
                    # Initialize the global variables to store the latent activations and relevances
                    latent_relevances_get[kwargs["layer_name"]] = None
                    latent_activations_get[kwargs["layer_name"]] = None
                    # register hooks
                    handles = register_hooks(
                        modified_model,
                        kwargs["layer_name"],
                        **{
                            "hooks": ["get_forward", "get_backward"],
                            "register_composite": None,
                        },
                    )

            output = modified_model(inputs)
            (relevance,) = torch.autograd.grad(
                outputs=output,
                inputs=inputs,
                grad_outputs=one_hot_max(output, targets).to(device),
                retain_graph=False,
                create_graph=False,
            )

            # Remove the handles if kwargs exists
            if bool(kwargs):
                if kwargs["method"] in ["latent_flipping", "latent_insertion"]:
                    [handle.remove() for handle in handles]

            # detach the relevance to avoid memory leak
            relevance = relevance.detach()

        return relevance



def flip_pixel(input, sorted_indices, steps, reference_point, **kwargs):
    """
    Flip the given image at the sorted indices with the reference value

    Args:
        input (torch.tensor): original image
        sorted_indices (torch.tensor): indices of the sorted heatmap
        steps (int): steps to be perturbed
        reference (str): reference value

    Returns:
        torch.tensor: perturbed image
    """
    batch_size = input.shape[0]
    num_channels = input.shape[1]
    batch_indices = torch.arange(batch_size).unsqueeze(1)
    # Replace the pixels at the sorted indices with the reference value
    input.view(batch_size, num_channels, -1)[
        batch_indices, :, sorted_indices[:, :steps]
    ] = reference_point.view(batch_size, num_channels, -1)[
        batch_indices, :, sorted_indices[:, :steps]
    ]

    return input



# The normal running script:
# python -m evaluations.pixel_perturbation --configs_path="evaluations/configs_perturbation.yaml"


@click.command()
@click.option("--configs_path")
def start(configs_path):
    evaluate(configs_path, "./results")  # for running on local machine


def evaluate(configs_path, dataset_path, output_path_parent, checkpoint_path):

    with open(configs_path, "r") as stream:
        configs = yaml.safe_load(stream)

    initialize_random_seed(configs["random_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _ = get_model_transforms( # model loader
        configs["architecture"], checkpoint_path, device=device
    )

    # Drop_last=True to avoid the last batch not to be of the same size
    dataset = get_dataset(configs["dataset"])(dataset_path=dataset_path)
    validation_dataloader = dataset.get_validation(
        batch_size=configs["batch_size"], drop_last=True, shuffle=True
    )

    if configs["attributor"] == "lrp":
        composite = get_composite(configs["composite_structure"])


    # Empty tensors to store the results
    logits_tensor = torch.tensor([]).to(device)
    acc_tensor = torch.tensor([]).to(device)
    targets_tensor = torch.tensor([]).to(device)
    batch_iteration_counter = 0
    for batch in tqdm.tqdm(validation_dataloader):
        if configs["stop_condition"] == batch_iteration_counter:
            print("Stop condition reached!")
            break
        batch_iteration_counter += 1

        # show the batch nr in tqdm
        tqdm.tqdm.write(f"Batch nr: {batch_iteration_counter}")

        x = batch[0].to(device)
        y = batch[1].to(device)
        # vanilla_outputs = model(x).detach()

        if configs["explainer"] == "random":
            heatmap = torch.randn(x.shape[0], x.shape[2], x.shape[3]).to(device)
        elif configs["explainer"] == "lrp":
            heatmap = lrp_attributor(
                model, x, y, composite, device, **configs
            ).sum(dim=1)


        reference_tensor = get_reference_tensor(configs["reference_value"], x.shape).to(
            device
        )

        with torch.no_grad():
            # Initialize some variables for further use
            image_dimension = (x.shape[2], x.shape[3])
            perturbation_input = x
            dimensions_multiplication = image_dimension[0] * image_dimension[1]


            # Compute the number of features(pixels) to be perturbed in each step
            features_in_step = int(dimensions_multiplication / configs["num_steps"])
            loop_tuple = (
                # features_in_step,
                0,  # TODO: check if this is correct
                # dimensions_multiplication,
                dimensions_multiplication + 1,  # TODO: check if this is correct
                features_in_step,
            )

            # Sort the heatmap in descending order used in the pixel perturbation(Flipping/Insertion)
            _, sorted_indices = torch.sort(
                heatmap.reshape(configs["batch_size"], -1),
                descending=configs["most_relevant_first"], # e.g., True enables flipping most relevant pixels first
            )


            # Empty tensor to store the outputs of the batch
            batch_outputs = torch.tensor([]).to(device)

            for step in range(*loop_tuple):
                perturbation_output = flip_pixel(
                    perturbation_input,
                    sorted_indices,
                    step,
                    reference_tensor,
                )

                # Compute the output of the model using the perturbed image(batch)
                output = model(
                    # perturbation_output.view(
                    #     -1, 3, image_dimension[0], image_dimension[1]
                    # )
                    perturbation_output  # TODO: check if this is correct
                )

                batch_outputs = torch.cat((batch_outputs, output.unsqueeze(1)), dim=-2)

        # remove the composite
        composite.remove()

        batch_parsed_logits, batch_parsed_acc, batch_target = parse_results_at_batch(
            batch_outputs, y
        )
        # Append results
        logits_tensor = torch.cat(
            (logits_tensor, batch_parsed_logits.unsqueeze(0)), dim=0
        )
        acc_tensor = torch.cat((acc_tensor, batch_parsed_acc.unsqueeze(0)), dim=0)
        targets_tensor = torch.cat((targets_tensor, batch_target.unsqueeze(0)), dim=0)


    save_results(module_name)(
        [acc_tensor, logits_tensor, targets_tensor],
        output_path_parent,
        f"{experiment_name}_{configs['composite_name']}_{configs['most_relevant_first']}",
    )


def get_reference_tensor(mode, dimension):
    """
    Get the reference value for the pixel perturbation
    This is the value that is going to be replaced with the original image
    in the pixel perturbation(Flipping/Insertion)

    Args:
        mode (Union["black", "gaussian"]): perturbation mode
        dimension (tuple): dimension of the given batch/image

    Returns:
        torch.tensor: the reference point(with dim of image/batch)
    """
    if mode == "black":
        valid_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )

        # Get the black image transformed using the validation transforms
        # It is not neccessarily equal to 0 in a transformed image
        black_image = valid_transforms(torch.zeros(dimension))
        return black_image

    elif mode == "gaussian":
        # Generate a random noise from a gaussian distribution
        return torch.randn(dimension)

    elif mode == "zero":
        # Generate a zero tensor(used in latent flipping/insertion)
        return torch.zeros(dimension)


def parse_results_at_batch(batch_logits, batch_targets):
    """
    Parse the batch logits and targets to compute the accuracy

    Args:
        batch_logits (torch.tensor): logits of the batch
        batch_targets (torch.tensor): targets of the batch

    Returns:
        float: accuracy of the batch
    """
    batch_indices = torch.arange(batch_targets.shape[0]).unsqueeze(1)

    softmax_tensor_batch = torch.softmax(batch_logits, dim=-1)
    softmax_res_batch = softmax_tensor_batch[
        batch_indices, :, batch_targets.type(torch.long).unsqueeze(1)
    ].squeeze()

    logit_res_bacth = batch_logits[
        batch_indices, :, batch_targets.type(torch.long).unsqueeze(1)
    ].squeeze()

    return logit_res_bacth, softmax_res_batch, batch_targets


if __name__ == "__main__":
    start()
