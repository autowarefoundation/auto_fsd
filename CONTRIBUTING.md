# Thank you for your interest in contributing! 
Autoware is supported by people like you, and all types and sizes of contribution are welcome. As a contributor, here are the guidelines that we would like you to follow for Autoware and its associated repositories.


# AI-assisted contributions and AI slop

We encourage contributors to use AI for coding, writing, translation, research,
and other parts of the development process. AI use does not need to be
disclosed. The contributor who submits an issue, pull request, review, or
comment remains responsible for its accuracy, relevance, quality, licensing,
and verification.

In this project, **AI slop** means unverified or low-quality content submitted
without reasonable effort to understand, check, and refine it, when doing so
creates more work for maintainers than value for the project. This standard
applies regardless of whether the content was produced by AI or by a person.
AI-like writing style or the output of an AI detector is not evidence of AI
slop.

Issues, pull requests, reviews, and comments must be specific to this
repository and must make a clear, technically grounded contribution. Examples
of unacceptable content include:

- Generic advice, restatements, or speculative recommendations that do not
  engage with the current code or discussion
- Repetitive, fragmented, off-topic, or bulk submissions
- Invented APIs, results, citations, behavior, or other claims that have not
  been checked
- Claims that tests, training, or other validation were run when they were not
- Code submitted without relevant validation or an honest statement of what
  was and was not tested
- Repeated failure to answer concrete maintainer questions about a submission

Disagreement, an unsuccessful experiment, or a documented regression is not AI
slop. Contributors should report limitations and negative results honestly.

## Validation of code changes

Contributors must make a reasonable effort to show that changed code works and
is likely to advance the stated goal. Describe the checks that were run, their
results, and any important limitations. We do not prescribe one execution
environment or validation procedure for every contribution; the evidence
should be appropriate for the change and sufficient for maintainers to assess
it.

Changes that can affect learned model behavior require stronger evidence. This
includes changes to model architecture, fusion, planners, losses, training
objectives, model inputs or targets, sampling, and dataset splits. Before such
a pull request is marked ready for review, it must include results from
training, report at least post-training ADE and FDE, and discuss what the
results indicate about the change, including regressions, trade-offs, and
limitations. A draft pull request may be opened while this validation is still
in progress.

Changes that do not affect learned model behavior, such as documentation, UI,
CI, deployment, observability, or supporting pipeline changes, do not require
ADE or FDE. They still require relevant tests or other evidence that the
changed behavior works. A data or training pipeline change that can alter
learned model behavior is treated as a model-affecting change.

## Moderation

Maintainers may close, lock, hide, or decline to review content that does not
meet these standards, without providing a detailed review. Repeated or serious
disruptive submissions may result in restrictions on further participation and
may be escalated to the Autoware Foundation or reported to GitHub when the
conduct also violates the applicable community or platform policies.

# Code of Conduct
To ensure the Autoware community stays open and inclusive, please follow the [Code of Conduct](https://github.com/autowarefoundation/autoware.privately-owned-vehicles/blob/main/CODE_OF_CONDUCT.md).

If you believe that someone in the community has violated the Code of Conduct, please make a report by emailing conduct@autoware.org

# What should I know before I get started?

## About this project

Please ensure that you have read the [README](https://github.com/autowarefoundation/autoware.privately-owned-vehicles/blob/main/README.md) file and developer [ONBOARDING](https://github.com/autowarefoundation/autoware.privately-owned-vehicles/blob/main/ONBOARDING.md) guide of this project to understand the goals and objectives and processes followed in this project.


## Contributing to open source projects
If you are new to open source projects, we recommend reading [GitHub's How to Contribute to Open Source](https://opensource.guide/how-to-contribute) guide for an overview of why people contribute to open source projects, what it means to contribute and much more besides.

# How can I get help?
Do not open issues for general support questions as we want to keep GitHub issues for confirmed bug reports. Instead, open a discussion in the Q&A category. For more details on the support mechanisms for Autoware, refer to the Support guidelines.

## Note

Issues created for questions or unconfirmed bugs will be moved to GitHub discussions by the maintainers.

# How can I contribute?

## Discussions
You can contribute to Autoware by facilitating and participating in [discussions](https://github.com/orgs/autowarefoundation/discussions), such as:

- Proposing a new feature to enhance Autoware
- Joining an existing discussion and expressing your opinion
- Organizing discussions for other contributors
- Answering questions and supporting other contributors

## Join and Participate in the Robotaxi Working group
To find out more, please read the [Onboarding Guide](./ONBOARDING.md)

## Bug reports
Before you report a bug, please search the issue tracker for the appropriate repository. It is possible that someone has already reported the same issue and that workarounds exist. If you can't determine the appropriate repository, ask the maintainers for help by creating a new discussion in the [Q&A category](https://github.com/autowarefoundation/autoware/discussions/new?category=q-a).

When reporting a bug, you should provide a minimal set of instructions to reproduce the issue. Doing so allows us to quickly confirm and focus on the right problem.

If you want to fix the bug by yourself that will be appreciated, but you should discuss possible approaches with the maintainers in the issue before submitting a pull request.

[Creating an issue is straightforward](https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue#creating-an-issue-from-a-repository), but if you happen to experience any problems then create a Q&A discussion to ask for help.

## Pull requests
You can submit pull requests for small changes such as:

- Minor documentation updates
- Fixing spelling mistakes
- Fixing CI failures
- Fixing warnings detected by compilers or analysis tools
- Making small changes to a single package
- If your pull request is a large change, the following process should be followed:

  1 - [Create a GitHub Discussion](https://docs.github.com/en/discussions/collaborating-with-your-community-using-discussions/collaborating-with-maintainers-using-discussions) to propose the change. Doing so allows you to get feedback from other members and the Autoware maintainers and to ensure that the proposed change is in line with Autoware's design philosophy and current development plans. If you're not sure where to have that conversation, then [create a new Q&A discussion](https://github.com/autowarefoundation/autoware/discussions/new?category=q-a).

  2 - [Create an issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue) following consensus in the discussions

  3 - [Create a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) to implement the changes that references the Issue created in step 2

  4 - Create documentation for the new addition (if relevant)

Examples of large changes include:

Adding a new feature to Autoware
Adding a new documentation page or section
For more information on how to submit a good pull request, have a read of the [pull request guidelines](https://autowarefoundation.github.io/autoware-documentation/main/contributing/pull-request-guidelines/) and don't forget to review the required [license notations](https://autowarefoundation.github.io/autoware-documentation/main/contributing/license/)!

## Pull request checks

We have various CI workflow checks to ensure the quality of pull requests.

### `semantic-pull-request`

This workflow makes sure [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) is applied to the pull request title.
Find more details in the [Autoware Documentation](https://autowarefoundation.github.io/autoware-documentation/main/contributing/pull-request-guidelines/#apply-conventional-commits-to-the-pull-request-title-required-automated).

### `pre-commit`

[pre-commit](https://pre-commit.ci/) is a tool to run formatters or linters when you commit.
This workflow checks whether the pull request has no error with pre-commit.

Find more info in the [Autoware Documentation](https://autowarefoundation.github.io/autoware-documentation/main/contributing/pull-request-guidelines/ci-checks/#pre-commit).
